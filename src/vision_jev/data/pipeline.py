"""Canonical question builders and source adapters.

Adapters write one JSON object per complete question and never expand candidates
into independent training rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

YES = {"yes", "true"}
NO = {"no", "false"}


def normalize_answer(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def stable_negatives(answer: str, pool: Iterable[str], key: str, count: int = 3) -> list[str]:
    candidates = sorted({normalize_answer(item) for item in pool} - {answer})
    if len(candidates) > 512:
        candidates = sorted(
            candidates,
            key=lambda item: hashlib.sha256(f"pool\0{item}".encode()).digest(),
        )[:512]
    ranked = sorted(
        candidates,
        key=lambda item: hashlib.sha256(f"{key}\0{item}".encode()).digest(),
    )
    return ranked[:count]


def compact_answer_pool(values: Iterable[str], limit: int = 512) -> list[str]:
    unique = {normalize_answer(value) for value in values}
    return sorted(
        unique,
        key=lambda item: hashlib.sha256(f"pool\0{item}".encode()).digest(),
    )[:limit]


def open_qa_sample(
    *,
    source: str,
    source_version: str,
    split: str,
    sample_id: str,
    group_id: str,
    image: str | None,
    question: str,
    answer: str,
    answer_pool: Iterable[str],
    license_name: str,
    evidence_reference: str,
    source_bucket: str = "public",
    label_origin: str = "human",
    origin_label_method: str = "human",
    quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_answer(answer)
    common = {
        "schema_version": 2,
        "sample_id": sample_id,
        "root_id": group_id,
        "group_id": group_id,
        "source": source,
        "source_version": source_version,
        "source_bucket": source_bucket,
        "license": license_name,
        "split": split,
        "image": image,
        "image_metadata": {"transform": "none", "visual_budget": "source_native"},
        "state_text": "",
        "question": question,
        "allowed_history": [],
        "input_track": "pure_visual" if image else "text_only",
        "label_origin": label_origin,
        "origin_label_method": origin_label_method,
        "language": "en",
        "evidence_reference": evidence_reference,
        "generator_version": "canonical-v1",
        "generator_revision": "canonical-v2",
        "proposal_kind": "none",
        "quality": dict(quality or {}),
        "teacher_only": False,
    }
    if normalized in YES | NO:
        return {
            **common,
            "task_type": "noul",
            "options": [],
            "candidates": [],
            "target_kind": "binary",
            "target": normalized in YES,
        }
    negatives = stable_negatives(normalized, answer_pool, sample_id)
    option_values = [normalized, *negatives]
    option_values.sort(key=lambda item: hashlib.sha256(f"{sample_id}:{item}".encode()).digest())
    options = [
        {"id": f"option_{index}", "text": value} for index, value in enumerate(option_values)
    ]
    target = next(option["id"] for option in options if option["text"] == normalized)
    return {
        **common,
        "task_type": "choice",
        "options": options,
        "candidates": options,
        "target_kind": "single",
        "target": target,
    }


def _write_jsonl(samples: Iterable[Mapping[str, Any]], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    from vision_jev.data.schema import validate_jsonl

    validate_jsonl(temporary, check_assets=True)
    temporary.replace(destination)
    return count


def _majority(answers: list[dict[str, Any]]) -> str:
    counts = Counter(normalize_answer(item["answer"]) for item in answers)
    return counts.most_common(1)[0][0]


def build_model_input(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize only fields available at inference time.

    This explicit allowlist prevents targets, future observations, verifier output,
    teacher traces, and evidence indexes from entering a model prompt.
    """
    allowlist = (
        "image",
        "image_metadata",
        "state_text",
        "question",
        "allowed_history",
        "input_track",
        "task_type",
        "candidates",
    )
    return {key: sample[key] for key in allowlist if key in sample}


def _dom_candidate(raw: str, action_type: str) -> dict[str, Any] | None:
    """Convert a Mind2Web DOM candidate without copying target-only flags."""
    try:
        node = json.loads(raw)
        attributes = json.loads(node.get("attributes", "{}"))
    except (json.JSONDecodeError, TypeError):
        return None
    backend_id = str(node.get("backend_node_id") or attributes.get("backend_node_id") or "")
    box_text = attributes.get("bounding_box_rect", "")
    try:
        x, y, width, height = (float(value) for value in box_text.split(","))
    except (TypeError, ValueError):
        return None
    if not backend_id or width <= 0 or height <= 0:
        return None
    visible_text = next(
        (
            str(attributes[key]).strip()
            for key in ("aria_label", "title", "placeholder", "value", "id", "role", "class")
            if attributes.get(key)
        ),
        str(node.get("tag", "element")),
    )
    return {
        "id": f"dom:{backend_id}",
        "text": f"{node.get('tag', 'element')}: {visible_text}"[:512],
        "box": [x, y, x + width, y + height],
        "action_spec": {"type": action_type},
    }


def normalize_mind2web(data_root: Path, destination: Path) -> int:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    root = data_root / "raw" / "multimodal_mind2web" / "snapshots" / "dataset"
    files = sorted((root / "data").glob("train-*.parquet"))
    image_root = data_root / "processed" / "multimodal_mind2web" / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "action_uid",
        "operation",
        "pos_candidates",
        "neg_candidates",
        "website",
        "annotation_id",
        "confirmed_task",
        "screenshot",
        "action_reprs",
        "target_action_index",
    ]

    def samples() -> Iterator[dict[str, Any]]:
        seen: set[str] = set()
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=32):
                for item in batch.to_pylist():
                    action_uid = str(item["action_uid"])
                    if action_uid in seen:
                        continue
                    seen.add(action_uid)
                    try:
                        operation = json.loads(item["operation"])
                        action_type = str(operation["op"]).upper()
                        action_index = int(item["target_action_index"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
                    if action_type not in {"CLICK", "TYPE", "SELECT"}:
                        continue
                    positive = [
                        value
                        for raw in item["pos_candidates"]
                        if (value := _dom_candidate(raw, action_type)) is not None
                    ]
                    negatives = [
                        value
                        for raw in item["neg_candidates"]
                        if (value := _dom_candidate(raw, action_type)) is not None
                    ]
                    if not positive or not negatives:
                        continue
                    negatives.sort(
                        key=lambda value: hashlib.sha256(
                            f"{action_uid}\0{value['id']}".encode()
                        ).digest()
                    )
                    candidates_by_id = {
                        candidate["id"]: candidate for candidate in [*positive, *negatives[:31]]
                    }
                    candidates = list(candidates_by_id.values())
                    target_ids = [
                        candidate["id"]
                        for candidate in positive
                        if candidate["id"] in candidates_by_id
                    ]
                    screenshot = item["screenshot"] or {}
                    image_bytes = screenshot.get("bytes")
                    if not image_bytes:
                        continue
                    image_path = image_root / f"{action_uid}.jpg"
                    if not image_path.exists():
                        image_path.write_bytes(image_bytes)
                    history = list(item["action_reprs"] or [])[:action_index]
                    yield {
                        "schema_version": 2,
                        "sample_id": f"mind2web:{action_uid}",
                        "root_id": f"mind2web:{action_uid}",
                        "group_id": f"mind2web-trace:{item['annotation_id']}",
                        "source": "multimodal_mind2web",
                        "source_version": "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889",
                        "source_bucket": "public",
                        "license": "OpenRAIL/research-disclaimer",
                        "split": "train",
                        "task_type": "choice",
                        "image": str(image_path),
                        "image_metadata": {"transform": "none", "visual_budget": "source_native"},
                        "state_text": item["confirmed_task"],
                        "question": "Select the next interface element to act on.",
                        "allowed_history": history,
                        "input_track": "high_level_goal_visual_dom_candidates",
                        "options": candidates,
                        "candidates": candidates,
                        "target_kind": "single" if len(target_ids) == 1 else "multiple",
                        "target": target_ids[0] if len(target_ids) == 1 else target_ids,
                        "label_origin": "human",
                        "origin_label_method": "human_action_demonstration",
                        "language": "en",
                        "evidence_reference": f"action_uid:{action_uid}",
                        "generator_version": "mind2web-adapter-v2",
                        "generator_revision": "canonical-v2",
                        "proposal_kind": "dom",
                        "quality": {
                            "pre_action_screenshot": True,
                            "target_visible_box": True,
                            "future_fields_excluded": True,
                            "website": item["website"],
                        },
                        "teacher_only": False,
                    }

    return _write_jsonl(samples(), destination)


def normalize_refcoco(data_root: Path, destination: Path) -> int:
    import pyarrow.parquet as pq

    source_root = data_root / "raw" / "refcoco" / "snapshots"
    datasets = ("refcoco", "refcocoplus", "refcocog")
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for path in sorted((source_root / dataset / "data").glob("*.parquet")):
            upstream_split = path.name.split("-", 1)[0]
            if upstream_split not in {"train", "validation"}:
                continue
            split = "train" if upstream_split == "train" else "dev"
            columns = [
                "ref_id",
                "ann_id",
                "image_id",
                "sentences",
                "bbox",
                "image_path",
                "global_image_id",
            ]
            for batch in pq.ParquetFile(path).iter_batches(columns=columns):
                for item in batch.to_pylist():
                    rows.append({"dataset": dataset, "split": split, **item})

    regions_by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    split_by_image: dict[str, str] = {}
    for row in rows:
        if row["global_image_id"] not in split_by_image or row["split"] == "dev":
            split_by_image[row["global_image_id"]] = row["split"]
        box = [float(value) for value in row["bbox"]]
        if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
            continue
        candidate_id = f"region:{row['ann_id']}"
        regions_by_image[row["global_image_id"]][candidate_id] = {
            "id": candidate_id,
            "text": f"region {candidate_id}",
            "box": box,
        }

    def samples() -> Iterator[dict[str, Any]]:
        for row in rows:
            candidates = list(regions_by_image[row["global_image_id"]].values())
            target = f"region:{row['ann_id']}"
            if target not in {candidate["id"] for candidate in candidates} or len(candidates) < 2:
                continue
            candidates.sort(
                key=lambda candidate: (
                    candidate["id"] != target,
                    hashlib.sha256(f"{row['ref_id']}\0{candidate['id']}".encode()).digest(),
                )
            )
            candidates = candidates[:32]
            candidates.sort(key=lambda candidate: candidate["id"])
            image_name = Path(row["image_path"]).name
            image = (
                data_root
                / "raw"
                / "vqav2"
                / "extracted"
                / "coco_train2014"
                / "train2014"
                / image_name
            )
            for sentence in row["sentences"]:
                sentence_id = sentence["sent_id"]
                yield {
                    "schema_version": 2,
                    "sample_id": f"{row['dataset']}:{row['ref_id']}:{sentence_id}",
                    "root_id": f"{row['dataset']}:{row['ref_id']}:{sentence_id}",
                    "group_id": row["global_image_id"],
                    "source": row["dataset"],
                    "source_version": "pinned-hf-mirror",
                    "source_bucket": "public",
                    "license": "RefCOCO/COCO-upstream-terms",
                    "split": split_by_image[row["global_image_id"]],
                    "task_type": "choice",
                    "image": str(image),
                    "image_metadata": {"transform": "none", "visual_budget": "source_native"},
                    "state_text": "",
                    "question": f"Which region matches: {sentence['sent']}?",
                    "allowed_history": [],
                    "input_track": "oracle_region_candidates",
                    "options": candidates,
                    "candidates": candidates,
                    "target_kind": "single",
                    "target": target,
                    "label_origin": "human",
                    "origin_label_method": "human_referring_expression",
                    "language": "en",
                    "evidence_reference": f"ann_id:{row['ann_id']}",
                    "generator_version": "refcoco-oracle-adapter-v2",
                    "generator_revision": "canonical-v2",
                    "proposal_kind": "oracle",
                    "quality": {"track": "oracle", "detector_metrics_must_be_separate": True},
                    "teacher_only": False,
                }

    return _write_jsonl(samples(), destination)


def normalize_chartqa(data_root: Path, destination: Path) -> int:
    import pyarrow.parquet as pq

    root = data_root / "raw" / "chartqa" / "snapshots" / "dataset"
    files = sorted((root / "data").glob("*.parquet"))
    pools: dict[str, list[str]] = defaultdict(list)
    split_by_digest: dict[str, str] = {}
    for path in files:
        split = "train" if "train" in path.name else "dev"
        if "test" in path.name:
            continue
        for batch in pq.ParquetFile(path).iter_batches(columns=["label", "image"]):
            for item in batch.to_pylist():
                pools[split].append(normalize_answer(item["label"]))
                digest = hashlib.sha256(item["image"]).hexdigest()
                if digest not in split_by_digest or split == "dev":
                    split_by_digest[digest] = split
    compact_pools = {split: compact_answer_pool(values) for split, values in pools.items()}
    image_root = data_root / "processed" / "chartqa" / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    def samples() -> Iterator[dict[str, Any]]:
        for path in files:
            if "test" in path.name:
                continue
            row_index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for item in batch.to_pylist():
                    current_index = row_index
                    row_index += 1
                    digest = hashlib.sha256(item["image"]).hexdigest()
                    canonical_split = split_by_digest[digest]
                    image_path = image_root / f"{digest}.png"
                    if not image_path.exists():
                        image_path.write_bytes(item["image"])
                    row_key = hashlib.sha256(
                        f"{path.name}\0{current_index}\0{item['imgname']}\0"
                        f"{item['query']}\0{item['label']}".encode()
                    ).hexdigest()[:20]
                    yield open_qa_sample(
                        source="chartqa",
                        source_version="af8b6f5c08c95085271561c2a3f9d15f2b5a9031",
                        split=canonical_split,
                        sample_id=f"chartqa:{row_key}",
                        group_id=f"chartqa-image:{digest}",
                        image=str(image_path),
                        question=item["query"],
                        answer=item["label"],
                        answer_pool=compact_pools[canonical_split],
                        license_name="ChartQA/GPL-3.0-review-required",
                        evidence_reference=f"image:{item['imgname']}:row:{current_index}",
                        origin_label_method=(
                            "human" if str(item.get("type", "")).lower() == "human" else "augmented"
                        ),
                        label_origin=(
                            "human"
                            if str(item.get("type", "")).lower() == "human"
                            else "programmatic"
                        ),
                        quality={
                            "question_origin": item.get("type"),
                            "table_excluded_from_input": True,
                            "answer_requires_table_verification": True,
                        },
                    )

    return _write_jsonl(samples(), destination)


def _redacted_action(raw: bytes | str) -> dict[str, Any] | None:
    try:
        action = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    allowed = {"action_type", "direction"}
    return {key: action[key] for key in allowed if key in action}


def normalize_android_control(data_root: Path, destination: Path) -> int:
    from android_env.proto.a11y import (  # type: ignore[import-untyped]
        android_accessibility_forest_pb2,
    )
    from tfrecord import example_pb2  # type: ignore[import-untyped]
    from tfrecord.reader import tfrecord_iterator  # type: ignore[import-untyped]

    root = data_root / "raw" / "android_control" / "snapshots" / "raw_tfrecords"
    files = sorted(path for path in root.glob("android_control-*") if path.is_file())
    split_path = data_root / "raw" / "android_control" / "downloads" / "splits.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    episode_splits = {
        int(episode_id): split
        for split in ("train", "validation", "test")
        for episode_id in split_payload[split]
    }
    image_root = data_root / "processed" / "android_control" / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    direct_actions = {"navigate_back", "navigate_home", "wait"}

    def samples() -> Iterator[dict[str, Any]]:
        for path in files:
            records = tfrecord_iterator(str(path), compression_type="gzip")
            for raw_record in records:
                example = example_pb2.Example()
                example.ParseFromString(bytes(raw_record))
                features = example.features.feature
                episode_id = str(features["episode_id"].int64_list.value[0])
                upstream_split = episode_splits.get(int(episode_id))
                if upstream_split not in {"train", "validation"}:
                    continue
                canonical_split = "train" if upstream_split == "train" else "dev"
                goal = features["goal"].bytes_list.value[0].decode("utf-8")
                screenshots = list(features["screenshots"].bytes_list.value)
                trees = list(features["accessibility_trees"].bytes_list.value)
                raw_actions = list(features["actions"].bytes_list.value)
                actions = [_redacted_action(raw) for raw in raw_actions]
                for step, action in enumerate(actions):
                    if action is None or step >= len(screenshots) or step >= len(trees):
                        continue
                    action_type = str(action.get("action_type", ""))
                    if action_type not in {"click", "long_press", "scroll", *direct_actions}:
                        continue
                    forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
                    try:
                        forest.ParseFromString(bytes(trees[step]))
                    except Exception:
                        continue
                    candidates: list[dict[str, Any]] = []
                    target_ids: list[str] = []
                    original_action = json.loads(bytes(raw_actions[step]).decode())
                    for window in forest.windows:
                        for node in window.tree.nodes:
                            box = node.bounds_in_screen
                            if (
                                not node.is_visible_to_user
                                or box.right <= box.left
                                or box.bottom <= box.top
                                or not (node.is_clickable or node.is_long_clickable)
                            ):
                                continue
                            candidate = {
                                "id": f"node:{node.unique_id}",
                                "text": "visible screen element",
                                "box": [box.left, box.top, box.right, box.bottom],
                                "action_spec": {"type": action_type},
                            }
                            candidates.append(candidate)
                            if action_type in {"click", "long_press"}:
                                x, y = original_action.get("x"), original_action.get("y")
                                if (
                                    x is not None
                                    and y is not None
                                    and box.left <= x <= box.right
                                    and box.top <= y <= box.bottom
                                ):
                                    target_ids.append(str(candidate["id"]))
                    # Android accessibility forests can repeat the same logical node
                    # across windows. Canonical questions require unique option IDs.
                    candidates = list(
                        {str(candidate["id"]): candidate for candidate in candidates}.values()
                    )
                    global_candidates = [
                        {
                            "id": f"global:scroll:{direction}",
                            "text": f"scroll {direction}",
                            "action_spec": {"type": "scroll", "direction": direction},
                        }
                        for direction in ("up", "down", "left", "right")
                    ] + [
                        {
                            "id": f"global:{kind}",
                            "text": kind.replace("navigate_", ""),
                            "action_spec": {"type": kind},
                        }
                        for kind in sorted(direct_actions)
                    ]
                    if action_type == "scroll":
                        target_ids = [f"global:scroll:{original_action.get('direction')}"]
                    elif action_type in direct_actions:
                        target_ids = [f"global:{action_type}"]
                    if not target_ids:
                        continue
                    target_set = set(target_ids)
                    candidates.sort(
                        key=lambda candidate: (
                            candidate["id"] not in target_set,
                            hashlib.sha256(
                                f"{episode_id}:{step}:{candidate['id']}".encode()
                            ).digest(),
                        )
                    )
                    candidates = [*candidates[:25], *global_candidates]
                    candidate_ids = {candidate["id"] for candidate in candidates}
                    target_ids = [target for target in target_ids if target in candidate_ids]
                    if not target_ids:
                        continue
                    image_path = image_root / f"{episode_id}-{step:04d}.png"
                    if not image_path.exists():
                        image_path.write_bytes(bytes(screenshots[step]))
                    history = [value for value in actions[:step] if value is not None]
                    yield {
                        "schema_version": 2,
                        "sample_id": f"android-control:{episode_id}:{step}",
                        "root_id": f"android-control:{episode_id}:{step}",
                        "group_id": f"android-control-episode:{episode_id}",
                        "source": "android_control",
                        "source_version": "ca530e89dfc3b1a4c3d9bd30fd36be94536563b1",
                        "source_bucket": "public",
                        "license": "AndroidControl/Apache-2.0-mirror-review-required",
                        "split": canonical_split,
                        "task_type": "choice",
                        "image": str(image_path),
                        "image_metadata": {"transform": "none", "visual_budget": "source_native"},
                        "state_text": goal,
                        "question": "Select the next action for the high-level goal.",
                        "allowed_history": history,
                        "input_track": "high_level_goal_pure_visual",
                        "options": candidates,
                        "candidates": candidates,
                        "target_kind": "single" if len(target_ids) == 1 else "multiple",
                        "target": target_ids[0] if len(target_ids) == 1 else target_ids,
                        "label_origin": "human",
                        "origin_label_method": "human_action_demonstration",
                        "language": "en",
                        "evidence_reference": f"episode:{episode_id}:step:{step}",
                        "generator_version": "android-control-adapter-v2",
                        "generator_revision": "canonical-v2",
                        "proposal_kind": "accessibility_tree",
                        "quality": {
                            "low_level_instruction_excluded": True,
                            "accessibility_text_excluded": True,
                            "future_actions_excluded": True,
                            "official_split": upstream_split,
                        },
                        "teacher_only": False,
                    }

    return _write_jsonl(samples(), destination)


PII_PATTERN = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:\+?\d[\d .()-]{7,}\d)\b|password|passwd)",
    re.IGNORECASE,
)


def sanitize_weblinx_candidate_text(value: str) -> str:
    normalized = " ".join(value.split())[:256]
    if PII_PATTERN.search(normalized):
        return "sensitive text redacted"
    return normalized or "visible webpage element"


def normalize_weblinx(data_root: Path, destination: Path) -> int:
    import pandas as pd  # type: ignore[import-untyped]

    root = data_root / "raw" / "weblinx"
    asset_root = root / "snapshots" / "train_subset" / "demonstrations"
    selection_path = data_root / "_state" / "selections" / "weblinx-train-v2.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            "WebLINX subset manifest not found; run data-download-weblinx-subset first"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = {(item["demo"], int(item["turn"])): item for item in selection["rows"]}
    table = pd.read_csv(root / "files" / "train_index" / "data" / "train.csv").fillna("")

    def parse_candidates(raw: str, intent: str) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        for part in re.split(r"\n(?=\(uid = )", raw):
            uid_match = re.search(r"\(uid = ([^)]+)\)", part)
            box_match = re.search(
                r"\[\[bbox\]\]\s*x=([-\d.]+)\s+y=([-\d.]+)\s+"
                r"width=([-\d.]+)\s+height=([-\d.]+)",
                part,
            )
            if uid_match is None or box_match is None:
                continue
            x, y, width, height = (float(value) for value in box_match.groups())
            if width <= 0 or height <= 0:
                continue
            text_match = re.search(r"\[\[text\]\]\s*(.*?)\s*\[\[bbox\]\]", part)
            visible_text = sanitize_weblinx_candidate_text(
                text_match.group(1) if text_match else ""
            )
            parsed.append(
                {
                    "id": f"element:{uid_match.group(1)}",
                    "text": visible_text,
                    "box": [x, y, x + width, y + height],
                    "action_spec": {"type": intent},
                }
            )
        return parsed

    def samples() -> Iterator[dict[str, Any]]:
        for row_index, row in table.iterrows():
            demo_name = str(row["demo"])
            turn_index = int(row["turn"])
            selected_row = selected.get((demo_name, turn_index))
            if selected_row is None:
                continue
            task = " ".join(str(row["utterances"]).split())
            if (
                not task
                or PII_PATTERN.search(task)
                or PII_PATTERN.search(str(row["action_history"]))
            ):
                continue
            intent = str(selected_row["intent"])
            target = f"element:{selected_row['target_uid']}"
            candidates = parse_candidates(str(row["candidates"]), intent)
            if target not in {item["id"] for item in candidates}:
                continue
            candidates.sort(
                key=lambda candidate: (
                    candidate["id"] != target,
                    hashlib.sha256(f"{demo_name}:{turn_index}:{candidate['id']}".encode()).digest(),
                )
            )
            candidates = candidates[:32]
            image = asset_root / demo_name / "screenshots" / selected_row["screenshot"]
            if not image.exists():
                continue
            yield {
                "schema_version": 2,
                "sample_id": f"weblinx:{demo_name}:{turn_index}",
                "root_id": f"weblinx:{demo_name}:{turn_index}",
                "group_id": f"weblinx-demo:{demo_name}",
                "source": "weblinx",
                "source_version": "36ef9f79b43df50e25b7f3b68e5c9f6ccf4160e8",
                "source_bucket": "public",
                "license": "CC-BY-NC-SA-4.0/fair-use",
                "split": "train",
                "task_type": "choice",
                "image": str(image),
                "image_metadata": {"transform": "none", "visual_budget": "source_native"},
                "state_text": task,
                "question": "Select the next webpage element to act on.",
                "allowed_history": [],
                "input_track": "dialogue_visual_region_candidates",
                "options": candidates,
                "candidates": candidates,
                "target_kind": "single",
                "target": target,
                "label_origin": "human",
                "origin_label_method": "human_dialogue_navigation",
                "language": "en",
                "evidence_reference": f"demo:{demo_name}:turn:{turn_index}:row:{row_index}",
                "generator_version": "weblinx-adapter-v2",
                "generator_revision": "canonical-v2",
                "proposal_kind": "dom",
                "quality": {
                    "future_turns_excluded": True,
                    "text_pii_regex_passed": True,
                    "image_pii_review": "pending",
                    "release_blocked": True,
                },
                "teacher_only": False,
            }

    return _write_jsonl(samples(), destination)


def normalize_vqav2(data_root: Path, destination: Path) -> int:
    root = data_root / "raw" / "vqav2" / "extracted"
    rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        question_path = next((root / f"questions_{split}").glob("*.json"))
        annotation_path = next((root / f"annotations_{split}").glob("*.json"))
        questions = json.loads(question_path.read_text(encoding="utf-8"))["questions"]
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))["annotations"]
        by_id = {item["question_id"]: item for item in annotations}
        for question in questions:
            annotation = by_id[question["question_id"]]
            normalized_answers = [
                normalize_answer(item["answer"]) for item in annotation["answers"]
            ]
            agreement = Counter(normalized_answers).most_common(1)[0][1]
            if agreement < 8:
                continue
            rows.append(
                {
                    "split": "train" if split == "train" else "dev",
                    "id": str(question["question_id"]),
                    "image_id": int(question["image_id"]),
                    "question": question["question"],
                    "answer": _majority(annotation["answers"]),
                    "agreement": agreement,
                    "upstream": str(annotation["question_id"]),
                }
            )
    pools: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        pools[row["split"]].append(row["answer"])
    compact_pools = {split: compact_answer_pool(values) for split, values in pools.items()}

    def samples() -> Iterator[dict[str, Any]]:
        for row in rows:
            coco_split = "train2014" if row["split"] == "train" else "val2014"
            filename = f"COCO_{coco_split}_{row['image_id']:012d}.jpg"
            yield open_qa_sample(
                source="vqav2",
                source_version="v2",
                split=row["split"],
                sample_id=f"vqav2:{row['id']}",
                group_id=f"coco:{row['image_id']}",
                image=str(
                    data_root
                    / "raw"
                    / "vqav2"
                    / "extracted"
                    / f"coco_{coco_split}"
                    / coco_split
                    / filename
                ),
                question=row["question"],
                answer=row["answer"],
                answer_pool=compact_pools[row["split"]],
                license_name="VQAv2/COCO-upstream-terms",
                evidence_reference=f"question_id:{row['upstream']}",
                quality={"answer_agreement": row["agreement"], "minimum_required": 8},
            )

    return _write_jsonl(samples(), destination)


def normalize_textvqa(data_root: Path, destination: Path) -> int:
    raw = data_root / "raw" / "textvqa" / "downloads"
    rows: list[dict[str, Any]] = []
    for upstream_split, split in (("train", "train"), ("val", "dev")):
        path = raw / f"TextVQA_0.5.1_{upstream_split}.json"
        for item in json.loads(path.read_text(encoding="utf-8"))["data"]:
            normalized_answers = [normalize_answer(value) for value in item["answers"]]
            agreement = Counter(normalized_answers).most_common(1)[0][1]
            if agreement >= 8:
                rows.append({**item, "canonical_split": split, "agreement": agreement})
    answer_pool = compact_answer_pool(
        normalize_answer(Counter(item["answers"]).most_common(1)[0][0]) for item in rows
    )

    def samples() -> Iterator[dict[str, Any]]:
        for item in rows:
            answer = Counter(normalize_answer(value) for value in item["answers"]).most_common(1)[
                0
            ][0]
            image = (
                data_root
                / "raw"
                / "textvqa"
                / "extracted"
                / "train_val_images"
                / "train_images"
                / f"{item['image_id']}.jpg"
            )
            yield open_qa_sample(
                source="textvqa",
                source_version="0.5.1",
                split=item["canonical_split"],
                sample_id=f"textvqa:{item['question_id']}",
                group_id=f"openimages:{item['image_id']}",
                image=str(image),
                question=item["question"],
                answer=answer,
                answer_pool=answer_pool,
                license_name="CC-BY-4.0",
                evidence_reference=f"question_id:{item['question_id']}",
                quality={
                    "answer_agreement": item["agreement"],
                    "minimum_required": 8,
                    "ocr_excluded_from_input": True,
                },
            )

    return _write_jsonl(samples(), destination)


def normalize_clevr(data_root: Path, destination: Path) -> int:
    root = data_root / "raw" / "clevr" / "extracted" / "clevr_v1" / "CLEVR_v1.0"
    rows: list[dict[str, Any]] = []
    for upstream_split, split in (("train", "train"), ("val", "dev")):
        path = root / "questions" / f"CLEVR_{upstream_split}_questions.json"
        for item in json.loads(path.read_text(encoding="utf-8"))["questions"]:
            rows.append({**item, "upstream_split": upstream_split, "canonical_split": split})
    pools = compact_answer_pool(item["answer"] for item in rows)

    def samples() -> Iterator[dict[str, Any]]:
        for item in rows:
            yield open_qa_sample(
                source="clevr",
                source_version="1.0",
                split=item["canonical_split"],
                sample_id=f"clevr:{item['upstream_split']}:{item['question_index']}",
                group_id=f"clevr:{item['image_filename']}",
                image=str(root / "images" / item["upstream_split"] / item["image_filename"]),
                question=item["question"],
                answer=item["answer"],
                answer_pool=pools,
                license_name="CC-BY-4.0",
                evidence_reference=f"program:{item['question_index']}",
            )

    return _write_jsonl(samples(), destination)


def normalize_gqa(data_root: Path, destination: Path) -> int:
    root = data_root / "raw" / "gqa" / "extracted" / "questions"
    rows: list[dict[str, Any]] = []
    for upstream_split, split in (("train", "train"), ("val", "dev")):
        path = root / f"{upstream_split}_balanced_questions.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for question_id, item in payload.items():
            rows.append({"id": question_id, "split": split, **item})
    pools: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        pools[row["split"]].append(row["answer"])
    compact_pools = {split: compact_answer_pool(values) for split, values in pools.items()}

    def samples() -> Iterator[dict[str, Any]]:
        for row in rows:
            yield open_qa_sample(
                source="gqa",
                source_version="1.2-balanced",
                split=row["split"],
                sample_id=f"gqa:{row['id']}",
                group_id=f"gqa-image:{row['imageId']}",
                image=str(
                    data_root
                    / "raw"
                    / "gqa"
                    / "extracted"
                    / "images"
                    / "images"
                    / f"{row['imageId']}.jpg"
                ),
                question=row["question"],
                answer=row["answer"],
                answer_pool=compact_pools[row["split"]],
                license_name="GQA/upstream-image-terms",
                evidence_reference="semantic:"
                + json.dumps(row.get("semantic", []), separators=(",", ":")),
                label_origin="programmatic",
                origin_label_method="programmatic_from_scene_graph",
                quality={"balanced_split": True, "negative_requires_scene_graph_audit": True},
            )

    return _write_jsonl(samples(), destination)


def normalize_mnli(data_root: Path, destination: Path) -> int:
    import pyarrow.parquet as pq

    files = sorted((data_root / "raw" / "multi_nli" / "snapshots" / "dataset").glob("**/*.parquet"))
    labels = {0: "entailment", 1: "neutral / insufficient information", 2: "contradiction"}

    def samples() -> Iterator[dict[str, Any]]:
        group_splits: dict[str, str] = {}
        for path in files:
            split = "train" if "train" in path.name else "dev"
            row_index = 0
            for batch in pq.ParquetFile(path).iter_batches(
                columns=["pairID", "premise", "hypothesis", "label"]
            ):
                for item in batch.to_pylist():
                    current_index = row_index
                    row_index += 1
                    if item["label"] not in labels:
                        continue
                    group_id = f"mnli:{item['pairID']}"
                    previous_split = group_splits.setdefault(group_id, split)
                    if previous_split != split:
                        continue
                    options = [
                        {"id": f"label_{key}", "text": value} for key, value in labels.items()
                    ]
                    yield {
                        "schema_version": 2,
                        "sample_id": f"mnli:{path.stem}:{current_index}:{item['pairID']}",
                        "root_id": group_id,
                        "group_id": group_id,
                        "source": "multi_nli",
                        "source_version": "pinned-hf-da70db2",
                        "source_bucket": "public",
                        "license": "MNLI/mixed-upstream-terms",
                        "split": split,
                        "task_type": "choice",
                        "image": None,
                        "image_metadata": {"transform": "none", "visual_budget": "none"},
                        "state_text": item["premise"],
                        "question": item["hypothesis"],
                        "allowed_history": [],
                        "input_track": "text_only",
                        "options": options,
                        "candidates": options,
                        "target_kind": "single",
                        "target": f"label_{item['label']}",
                        "label_origin": "human",
                        "origin_label_method": "human_consensus",
                        "language": "en",
                        "evidence_reference": f"pairID:{item['pairID']}",
                        "generator_version": "mnli-v2",
                        "generator_revision": "canonical-v2",
                        "proposal_kind": "none",
                        "quality": {},
                        "teacher_only": False,
                    }

    return _write_jsonl(samples(), destination)


ADAPTERS = {
    "multimodal_mind2web": normalize_mind2web,
    "refcoco": normalize_refcoco,
    "chartqa": normalize_chartqa,
    "android_control": normalize_android_control,
    "weblinx": normalize_weblinx,
    "vqav2": normalize_vqav2,
    "gqa": normalize_gqa,
    "textvqa": normalize_textvqa,
    "clevr": normalize_clevr,
    "multi_nli": normalize_mnli,
}


def normalize_source(source: str, data_root: Path, destination: Path) -> int:
    try:
        adapter = ADAPTERS[source]
    except KeyError as exc:
        raise ValueError(
            f"normalizer not implemented for {source}; available: {sorted(ADAPTERS)}"
        ) from exc
    return adapter(data_root, destination)
