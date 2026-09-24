"""Resumable, provenance-aware dataset downloader."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WEBLINX_REVISION = "36ef9f79b43df50e25b7f3b68e5c9f6ccf4160e8"
GUI_ODYSSEY_REVISION = "71e0e7e2d169c642e7a99c264d82ffecadf67889"
_WEBLINX_ACTION = re.compile(r'^(click|submit)\(uid="([^"]+)"')


@dataclass(frozen=True)
class ArtifactResult:
    source: str
    artifact: str
    kind: str
    status: str
    path: str
    bytes: int
    sha256: str | None
    revision: str | None
    completed_at: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    marker = destination / ".vision_jev_extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    marker.write_text(f"archive_sha256={_sha256(archive)}\n", encoding="utf-8")


def _http_download(url: str, destination: Path, expected_bytes: int | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Vision-JEV/0.1 data-pipeline"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        ranged = response.status == 206
        content_length = response.headers.get("Content-Length")
        response_bytes = int(content_length) if content_length is not None else None
        if existing and not ranged:
            partial.unlink()
            existing = 0
        mode = "ab" if existing and ranged else "wb"
        expected_size = existing + response_bytes if response_bytes is not None else None
        downloaded = existing
        last_report = downloaded
        started = time.monotonic()
        with partial.open(mode) as handle:
            while block := response.read(8 * 1024 * 1024):
                handle.write(block)
                downloaded += len(block)
                if downloaded - last_report >= 256 * 1024 * 1024:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"download {destination.name}: {downloaded / 2**30:.2f} GiB "
                        f"({(downloaded - existing) / elapsed / 2**20:.1f} MiB/s)",
                        flush=True,
                    )
                    last_report = downloaded
    actual_size = partial.stat().st_size
    response_mismatch = expected_size is not None and actual_size != expected_size
    catalog_mismatch = expected_bytes is not None and actual_size != expected_bytes
    if response_mismatch or catalog_mismatch:
        expected = expected_bytes if expected_bytes is not None else expected_size
        raise OSError(
            f"truncated download for {destination}: got {actual_size} bytes, "
            f"expected {expected}; resumable partial retained"
        )
    partial.replace(destination)
    return destination


def _filename(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?", 1)[0]


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {source["id"]: source for source in payload["sources"]}


def _quarantine_invalid_archive(path: Path) -> Path | None:
    """Move an invalid ZIP aside instead of deleting user data."""
    if not path.exists() or zipfile.is_zipfile(path):
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_name(f"{path.name}.invalid-{stamp}")
    path.replace(quarantine)
    print(f"quarantined invalid archive: {quarantine}", flush=True)
    return quarantine


def download_source(
    source: dict[str, Any], data_root: Path, extract: bool = True
) -> list[ArtifactResult]:
    source_root = data_root / "raw" / source["id"]
    results: list[ArtifactResult] = []
    token = os.environ.get("HF_TOKEN")
    for artifact in source["artifacts"]:
        print(f"[{source['id']}] {artifact['name']} ({artifact['kind']})", flush=True)
        if artifact["kind"] == "http":
            archive = source_root / "downloads" / _filename(artifact["url"])
            if artifact.get("extract", False):
                _quarantine_invalid_archive(archive)
            if not archive.exists():
                for attempt in range(1, 9):
                    try:
                        _http_download(artifact["url"], archive, artifact.get("bytes"))
                        break
                    except OSError:
                        if attempt == 8:
                            raise
                        delay = min(2**attempt, 30)
                        print(
                            f"download interrupted; retry {attempt}/8 in {delay}s "
                            f"(partial retained)",
                            flush=True,
                        )
                        time.sleep(delay)
            expected_bytes = artifact.get("bytes")
            if expected_bytes is not None and archive.stat().st_size != expected_bytes:
                raise OSError(
                    f"size mismatch for {archive}: got {archive.stat().st_size}, "
                    f"expected {expected_bytes}"
                )
            digest = _sha256(archive)
            expected = artifact.get("sha256")
            if expected and digest != expected:
                raise ValueError(f"checksum mismatch for {archive}: {digest} != {expected}")
            if extract and artifact.get("extract", False):
                _safe_extract(archive, source_root / "extracted" / artifact["name"])
            result_path = archive
            revision = None
        elif artifact["kind"] == "hf_snapshot":
            from huggingface_hub import snapshot_download

            target = source_root / "snapshots" / artifact["name"]
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=artifact["repo_id"],
                repo_type="dataset",
                revision=artifact["revision"],
                local_dir=target,
                token=token,
                allow_patterns=artifact.get("allow_patterns"),
                max_workers=int(artifact.get("max_workers", 8)),
            )
            result_path = target
            digest = None
            revision = artifact["revision"]
        elif artifact["kind"] == "hf_file":
            from huggingface_hub import hf_hub_download

            target = source_root / "files" / artifact["name"]
            target.mkdir(parents=True, exist_ok=True)
            downloaded = Path(
                hf_hub_download(
                    repo_id=artifact["repo_id"],
                    repo_type="dataset",
                    filename=artifact["filename"],
                    revision=artifact["revision"],
                    local_dir=target,
                    token=token,
                )
            )
            result_path = downloaded
            digest = _sha256(downloaded)
            revision = artifact["revision"]
        else:
            raise ValueError(f"unsupported artifact kind: {artifact['kind']}")
        size = (
            result_path.stat().st_size
            if result_path.is_file()
            else sum(item.stat().st_size for item in result_path.rglob("*") if item.is_file())
        )
        results.append(
            ArtifactResult(
                source=source["id"],
                artifact=artifact["name"],
                kind=artifact["kind"],
                status="complete",
                path=str(result_path),
                bytes=size,
                sha256=digest,
                revision=revision,
                completed_at=datetime.now(UTC).isoformat(),
            )
        )
    state = data_root / "_state" / "downloads"
    state.mkdir(parents=True, exist_ok=True)
    temporary = state / f"{source['id']}.json.tmp"
    final = state / f"{source['id']}.json"
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": source["id"],
                "homepage": source["homepage"],
                "license_status": source["license_status"],
                "artifacts": [asdict(item) for item in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(final)
    return results


def inventory(data_root: Path) -> dict[str, Any]:
    states = data_root / "_state" / "downloads"
    records = []
    if states.exists():
        records = [
            json.loads(path.read_text(encoding="utf-8")) for path in sorted(states.glob("*.json"))
        ]
    return {
        "data_root": str(data_root),
        "downloaded_sources": [record["source"] for record in records],
        "bytes": sum(
            artifact["bytes"] for record in records for artifact in record.get("artifacts", [])
        ),
        "records": records,
        "free_bytes": shutil.disk_usage(data_root).free if data_root.exists() else None,
    }


def materialize_gui_odyssey_subset(
    data_root: Path,
    *,
    target_rows: int = 8500,
    seed: str = "vision-jev-sft",
    max_workers: int = 8,
) -> dict[str, Any]:
    """Select train steps first, then fetch only their source screenshots."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    source_root = data_root / "raw" / "gui_odyssey"
    annotations_path = source_root / "files" / "annotations" / "all_anno.json"
    split_path = source_root / "files" / "random_split" / "splits" / "random_split.json"
    if not annotations_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("download GUI-Odyssey annotations and random_split first")
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = {Path(value).stem for value in split_payload["train"]}
    eligible: list[dict[str, Any]] = []
    allowed_actions = {"CLICK", "SCROLL", "LONG_PRESS", "TEXT", "COMPLETE", "IMPOSSIBLE"}
    for episode in annotations:
        episode_id = str(episode["episode_id"])
        if episode_id not in train_ids:
            continue
        steps = ast.literal_eval(str(episode["steps"]))
        for step in steps:
            if step.get("action") not in allowed_actions:
                continue
            key = f"{seed}\0{episode_id}\0{step['step']}"
            eligible.append(
                {
                    "rank": hashlib.sha256(key.encode()).hexdigest(),
                    "episode_id": episode_id,
                    "device_name": episode.get("device_name"),
                    "category": episode.get("category"),
                    "task": episode.get("task"),
                    "instruction": episode.get("instruction"),
                    "step": int(step["step"]),
                    "screenshot": str(step["screenshot"]),
                    "action": str(step["action"]),
                    "info": step.get("info"),
                    "history": [
                        {"action": previous.get("action"), "info": previous.get("info")}
                        for previous in steps[: int(step["step"])]
                    ],
                }
            )
    selected = sorted(eligible, key=lambda item: item["rank"])[:target_rows]
    if len(selected) < target_rows:
        raise ValueError(f"GUI-Odyssey has only {len(selected)} eligible train steps")
    target = source_root / "selected"
    token = os.environ.get("HF_TOKEN")
    metadata_root = source_root / "_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    screenshot_index_path = metadata_root / "screenshot-paths.json"
    screenshot_paths: dict[str, str]
    if screenshot_index_path.is_file():
        screenshot_paths = json.loads(screenshot_index_path.read_text(encoding="utf-8"))
    else:
        screenshot_paths = {}
        for entry in list_repo_tree(
            repo_id="OpenGVLab/GUI-Odyssey",
            repo_type="dataset",
            path_in_repo="screenshots",
            recursive=True,
            revision=GUI_ODYSSEY_REVISION,
            token=token,
        ):
            path = str(entry.path)
            if path.endswith(".png"):
                screenshot_paths[Path(path).name] = path
        temporary_index = screenshot_index_path.with_suffix(".json.tmp")
        temporary_index.write_text(
            json.dumps(screenshot_paths, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_index.replace(screenshot_index_path)
    missing_paths = sorted(
        item["screenshot"] for item in selected if item["screenshot"] not in screenshot_paths
    )
    if missing_paths:
        raise FileNotFoundError(f"GUI-Odyssey screenshot index misses {missing_paths[:3]}")

    def fetch(item: dict[str, Any]) -> str:
        filename = screenshot_paths[item["screenshot"]]
        return hf_hub_download(
            repo_id="OpenGVLab/GUI-Odyssey",
            repo_type="dataset",
            filename=filename,
            revision=GUI_ODYSSEY_REVISION,
            local_dir=target,
            token=token,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, item): item for item in selected}
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 250 == 0 or completed == len(selected):
                print(f"GUI-Odyssey screenshots: {completed}/{len(selected)}", flush=True)
    index_path = target / "index.jsonl"
    temporary = index_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in selected:
            clean = {key: value for key, value in item.items() if key != "rank"}
            clean["image"] = str(target / screenshot_paths[item["screenshot"]])
            handle.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(index_path)
    return {
        "revision": GUI_ODYSSEY_REVISION,
        "seed": seed,
        "eligible_train_steps": len(eligible),
        "selected_steps": len(selected),
        "index": str(index_path),
    }


def materialize_weblinx_subset(
    data_root: Path, *, target_rows: int = 7000, seed: str = "vision-jev-sft-v2"
) -> dict[str, Any]:
    """Fetch only WebLINX train screenshots needed for a deterministic safe subset."""
    import pandas as pd  # type: ignore[import-untyped]
    from huggingface_hub import hf_hub_download

    source_root = data_root / "raw" / "weblinx"
    csv_path = source_root / "files" / "train_index" / "data" / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError("download the WebLINX train_index artifact first")
    rows = pd.read_csv(csv_path).fillna("")
    eligible: list[dict[str, Any]] = []
    for row_index, row in rows.iterrows():
        match = _WEBLINX_ACTION.match(str(row["action"]))
        if match is None:
            continue
        target_uid = match.group(2)
        candidates = str(row["candidates"])
        if f"uid = {target_uid}" not in candidates or "[[bbox]]" not in candidates:
            continue
        eligible.append(
            {
                "row_index": int(row_index),
                "demo": str(row["demo"]),
                "turn": int(row["turn"]),
                "intent": match.group(1),
                "target_uid": target_uid,
            }
        )

    # Download replay metadata first: it is the authoritative turn-to-screenshot mapping.
    demo_names = sorted({item["demo"] for item in eligible})
    target = source_root / "snapshots" / "train_subset"
    token = os.environ.get("HF_TOKEN")

    def fetch_exact(filenames: list[str], stage: str) -> None:
        def fetch(filename: str) -> str:
            if (target / filename).is_file():
                return filename
            hf_hub_download(
                repo_id="McGill-NLP/WebLINX-full",
                repo_type="dataset",
                filename=filename,
                revision=WEBLINX_REVISION,
                local_dir=target,
                token=token,
            )
            return filename

        completed = 0
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = [executor.submit(fetch, filename) for filename in filenames]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 250 == 0 or completed == len(filenames):
                    print(f"WebLINX {stage}: {completed}/{len(filenames)}", flush=True)

    fetch_exact(
        [f"demonstrations/{name}/replay.json" for name in demo_names],
        "replay metadata",
    )

    by_demo: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        by_demo.setdefault(str(item["demo"]), []).append(item)
    mapped: list[dict[str, Any]] = []
    for demo_name, demo_items in by_demo.items():
        replay_path = target / "demonstrations" / demo_name / "replay.json"
        payload = json.loads(replay_path.read_text(encoding="utf-8"))["data"]
        for item in demo_items:
            if item["turn"] < 0 or item["turn"] >= len(payload):
                continue
            turn = payload[item["turn"]]
            state = turn.get("state") or {}
            screenshot = state.get("screenshot")
            if not screenshot or state.get("screenshot_status") != "good":
                continue
            mapped.append({**item, "screenshot": str(screenshot)})

    mapped.sort(
        key=lambda item: hashlib.sha256(f"{seed}:{item['demo']}:{item['turn']}".encode()).digest()
    )
    selected = mapped[:target_rows]
    if len(selected) < target_rows:
        raise ValueError(
            f"WebLINX safe visual subset has {len(selected)} rows; requested {target_rows}"
        )
    screenshot_patterns = sorted(
        {f"demonstrations/{item['demo']}/screenshots/{item['screenshot']}" for item in selected}
    )
    fetch_exact(screenshot_patterns, "screenshots")

    selection_dir = data_root / "_state" / "selections"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_path = selection_dir / "weblinx-train-v2.json"
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": WEBLINX_REVISION,
                "seed": seed,
                "rows": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(selection_path)
    return {
        "eligible_rows": len(eligible),
        "visual_rows": len(mapped),
        "selected_rows": len(selected),
        "selected_demos": len({item["demo"] for item in selected}),
        "selection": str(selection_path),
        "asset_root": str(target),
    }
