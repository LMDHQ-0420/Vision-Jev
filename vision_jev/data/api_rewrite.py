"""API-assisted, same-language question rewrites with deterministic checks."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_jev.api_keys import APIKeys
from vision_jev.data.schema import validate_jsonl

PROVIDERS = {
    "kimi": {"base_url": "https://api.moonshot.cn/v1", "model": "kimi-k2.6"},
    "glm": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-5.3-flash"},
}
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?!\w)")
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|except|neither|nor|cannot|can't|isn't|aren't)\b", re.I
)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class RewriteResult:
    accepted: int
    attempted: int
    rejected: dict[str, int]
    provider: str
    model: str
    output: str


@dataclass(frozen=True)
class BatchRewriteResult:
    accepted: int
    requested: int
    rejected: dict[str, int]
    task_counts: dict[str, int]
    provider: str
    model: str
    batch_id: str
    output: str


def audit_rewrites(candidates: Path, parent_manifest: Path) -> dict[str, Any]:
    """Compare every derived row with its parent and report invariant violations."""
    rows = [json.loads(raw) for raw in candidates.read_text(encoding="utf-8").splitlines() if raw]
    parent_ids = {str(row.get("quality", {}).get("parent_sample_id", "")) for row in rows}
    parents: dict[str, dict[str, Any]] = {}
    with parent_manifest.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            sample = json.loads(raw)
            sample_id = str(sample["sample_id"])
            if sample_id in parent_ids:
                parents[sample_id] = sample
    immutable = (
        "root_id",
        "group_id",
        "source_version",
        "license",
        "split",
        "task_type",
        "image",
        "image_metadata",
        "state_text",
        "allowed_history",
        "input_track",
        "options",
        "candidates",
        "target_kind",
        "target",
        "label_origin",
        "language",
        "teacher_only",
    )
    violations: Counter[str] = Counter()
    lengths: list[float] = []
    sources: Counter[str] = Counter()
    for row in rows:
        parent_id = str(row.get("quality", {}).get("parent_sample_id", ""))
        parent = parents.get(parent_id)
        if parent is None:
            violations["missing_parent"] += 1
            continue
        sources[str(parent["source"])] += 1
        for field in immutable:
            if row.get(field) != parent.get(field):
                violations[f"changed_{field}"] += 1
        reason = validate_rewrite(
            str(parent["question"]), str(row["question"]), str(parent["language"])
        )
        if reason is not None:
            violations[f"rewrite_{reason}"] += 1
        lengths.append(len(str(row["question"])) / max(len(str(parent["question"])), 1))
    validation = validate_jsonl(candidates, check_assets=True)
    ordered_lengths = sorted(lengths)

    def percentile(fraction: float) -> float | None:
        if not ordered_lengths:
            return None
        index = round((len(ordered_lengths) - 1) * fraction)
        return round(ordered_lengths[index], 3)

    return {
        "path": str(candidates),
        "questions": validation.questions,
        "task_counts": validation.task_counts,
        "language_counts": dict(Counter(str(row["language"]) for row in rows)),
        "parent_source_counts": dict(sorted(sources.items())),
        "invariant_violations": dict(sorted(violations.items())),
        "length_ratio": {
            "min": percentile(0.0),
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "max": percentile(1.0),
        },
    }


def _normalized(text: str) -> str:
    return _SPACE.sub(" ", text.strip()).casefold()


def select_parents(
    manifest: Path, *, choice: int, noul: int, seed: str = "vision-jev-api-rewrite"
) -> list[dict[str, Any]]:
    """Select exact deterministic task quotas from train data."""
    wanted = {"choice": choice, "noul": noul}
    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = {key: [] for key in wanted}
    with manifest.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            sample = json.loads(raw)
            task = str(sample.get("task_type"))
            if task not in wanted or sample.get("split") != "train":
                continue
            rank = hashlib.sha256(f"{seed}\0{sample['sample_id']}".encode()).hexdigest()
            candidates[task].append((rank, sample))
    selected: list[dict[str, Any]] = []
    for task, count in wanted.items():
        ranked = sorted(candidates[task], key=lambda item: item[0])
        if len(ranked) < count:
            raise ValueError(f"only {len(ranked)} eligible {task} parents; need {count}")
        selected.extend(sample for _, sample in ranked[:count])
    return sorted(selected, key=lambda sample: str(sample["sample_id"]))


def rewrite_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    options = [str(option["text"]) for option in sample["options"]]
    payload = {
        "language": sample["language"],
        "state_text": sample["state_text"],
        "question": sample["question"],
        "options": options,
    }
    return [
        {
            "role": "system",
            "content": (
                "Substantially rephrase only the question while preserving its exact meaning. "
                "Change its wording or syntax; never return the original sentence verbatim. "
                "Use natural, concise wording common in dataset questions; avoid ornate, rare, "
                "overly formal, or vague synonyms. Never replace action or step with mission, "
                "course, measure, method, or operation. Correct obvious spelling mistakes. "
                "Use the same language. "
                "Preserve polarity, negation, quantities, "
                "named entities, relations, and temporal or spatial constraints. "
                "Do not answer, explain, add facts, or mention this instruction. "
                "Return one JSON object with exactly one string field named question."
            ),
        },
        {
            "role": "user",
            "content": "Rewrite this JSON input. Return JSON only:\n"
            + json.dumps(payload, ensure_ascii=False),
        },
    ]


def grouped_rewrite_messages(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    items = [
        {
            "id": str(sample["sample_id"]),
            "language": sample["language"],
            "state_text": sample["state_text"],
            "question": sample["question"],
            "options": [str(option["text"]) for option in sample["options"]],
        }
        for sample in samples
    ]
    return [
        {
            "role": "system",
            "content": (
                "For every input item, substantially rephrase only its question while preserving "
                "the exact meaning. Change wording or syntax and never copy the original verbatim. "
                "Use natural, concise wording common in dataset questions; avoid ornate, rare, "
                "overly formal, or vague synonyms. Never replace action or step with mission, "
                "course, measure, method, or operation. Correct obvious spelling mistakes. "
                "Use the same language. Preserve polarity, "
                "negation, quantities, named entities, relations, and temporal or spatial "
                "constraints. Do not answer, explain, add facts, or alter any id. Return one JSON "
                "object with exactly one field named rewrites. Its value must be an array "
                "containing "
                "exactly one object per input, each with exactly id and question string fields."
            ),
        },
        {
            "role": "user",
            "content": "Rewrite every item in this JSON array. Return JSON only:\n"
            + json.dumps(items, ensure_ascii=False),
        },
    ]


def validate_rewrite(original: str, rewritten: str, language: str) -> str | None:
    """Return a machine-readable rejection reason, or None when accepted."""
    before = original.strip()
    after = rewritten.strip()
    if not after:
        return "empty"
    if _normalized(before) == _normalized(after):
        return "unchanged"
    ratio = len(after) / max(len(before), 1)
    if ratio < 0.45 or ratio > 2.2:
        return "length_ratio"
    if Counter(_NUMBER.findall(before)) != Counter(_NUMBER.findall(after)):
        return "numbers_changed"
    if Counter(match.group(0).casefold() for match in _NEGATION.finditer(before)) != Counter(
        match.group(0).casefold() for match in _NEGATION.finditer(after)
    ):
        return "negation_changed"
    if language == "en":
        alphabetic = [char for char in after if char.isalpha()]
        latin = [char for char in alphabetic if "a" <= char.casefold() <= "z"]
        if alphabetic and len(latin) / len(alphabetic) < 0.8:
            return "language_changed"
    if language.startswith("zh"):
        han = re.findall(r"[\u3400-\u9fff]", after)
        if len(han) < 2:
            return "language_changed"
    return None


def make_rewrite(
    parent: dict[str, Any], question: str, provider: str, model: str
) -> dict[str, Any]:
    sample = deepcopy(parent)
    parent_id = str(parent["sample_id"])
    digest = hashlib.sha256(f"{provider}\0{model}\0{parent_id}\0{question}".encode()).hexdigest()[
        :16
    ]
    sample["sample_id"] = f"api-rewrite:{digest}"
    sample["source"] = "api_rewrite"
    sample["source_bucket"] = "api_assisted"
    sample["question"] = question.strip()
    sample["label_origin"] = parent["label_origin"]
    sample["origin_label_method"] = "inherited_verified_public_label"
    sample["generator_revision"] = f"{provider}:{model}"
    quality = dict(parent.get("quality", {}))
    quality.update(
        {
            "parent_sample_id": parent_id,
            "rewrite_provider": provider,
            "rewrite_model": model,
            "automatic_checks": "passed",
        }
    )
    sample["quality"] = quality
    return sample


def _extract_question(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    payload = json.loads(text)
    if set(payload) != {"question"} or not isinstance(payload["question"], str):
        raise ValueError("response must contain exactly one string field: question")
    return str(payload["question"])


def _extract_grouped_questions(content: str, expected_ids: set[str]) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    payload = json.loads(text)
    if set(payload) != {"rewrites"} or not isinstance(payload["rewrites"], list):
        raise ValueError("response must contain exactly one rewrites array")
    result: dict[str, str] = {}
    for item in payload["rewrites"]:
        if not isinstance(item, dict) or set(item) != {"id", "question"}:
            raise ValueError("each rewrite must contain exactly id and question")
        sample_id = str(item["id"])
        question = item["question"]
        if sample_id in result or not isinstance(question, str):
            raise ValueError("rewrite ids must be unique and questions must be strings")
        result[sample_id] = question
    if set(result) != expected_ids:
        raise ValueError("rewrite ids do not exactly match input ids")
    return result


def _client(provider: str, keys: APIKeys) -> tuple[Any, str]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise RuntimeError("install the data dependencies to use API rewriting") from exc
    if provider not in PROVIDERS:
        raise ValueError(f"unsupported provider: {provider}")
    config = PROVIDERS[provider]
    return (
        OpenAI(
            api_key=getattr(keys, provider),
            base_url=config["base_url"],
            max_retries=0,
        ),
        config["model"],
    )


def generate_rewrites(
    parents: Iterable[dict[str, Any]],
    *,
    keys: APIKeys,
    destination: Path,
    provider: str = "kimi",
    max_attempts: int = 3,
    max_workers: int = 4,
    group_size: int = 1,
    min_interval_seconds: float = 0.0,
) -> RewriteResult:
    """Generate a small or fallback batch synchronously and retain only checked rows."""
    client, model = _client(provider, keys)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_rows = list(parents)
    if group_size > 1:
        return _generate_grouped_rewrites(
            parent_rows,
            client=client,
            model=model,
            destination=destination,
            provider=provider,
            max_attempts=max_attempts,
            group_size=group_size,
            min_interval_seconds=min_interval_seconds,
            max_workers=max_workers,
        )

    def process(parent: dict[str, Any]) -> tuple[dict[str, Any] | None, int, Counter[str]]:
        local_rejected: Counter[str] = Counter()
        local_attempted = 0
        for attempt in range(max_attempts):
            local_attempted += 1
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=rewrite_messages(parent),
                    response_format={"type": "json_object"},
                    temperature=0.6 if provider == "kimi" else 0.3,
                    extra_body={"thinking": {"type": "disabled"}} if provider == "kimi" else {},
                    timeout=120,
                )
                content = response.choices[0].message.content or ""
                question = _extract_question(content)
                reason = validate_rewrite(
                    str(parent["question"]), question, str(parent["language"])
                )
                if reason is None:
                    return (
                        make_rewrite(parent, question, provider, model),
                        local_attempted,
                        local_rejected,
                    )
                local_rejected[reason] += 1
            except Exception as exc:  # provider errors are recorded without secret-bearing payloads
                local_rejected[type(exc).__name__] += 1
            if attempt + 1 < max_attempts:
                time.sleep(min(2**attempt, 4))
        local_rejected["exhausted_parent"] += 1
        return None, local_attempted, local_rejected

    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    attempted = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outcomes = executor.map(process, parent_rows)
        for sample, local_attempted, local_rejected in outcomes:
            attempted += local_attempted
            rejected.update(local_rejected)
            if sample is None:
                continue
            accepted.append(sample)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in accepted:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    if accepted:
        validate_jsonl(destination, check_assets=True)
    return RewriteResult(
        accepted=len(accepted),
        attempted=attempted,
        rejected=dict(sorted(rejected.items())),
        provider=provider,
        model=model,
        output=str(destination),
    )


def _generate_grouped_rewrites(
    parents: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    destination: Path,
    provider: str,
    max_attempts: int,
    group_size: int,
    min_interval_seconds: float,
    max_workers: int,
) -> RewriteResult:
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    accepted_parent_ids: set[str] = set()
    if destination.exists():
        for raw in destination.read_text(encoding="utf-8").splitlines():
            if not raw:
                continue
            sample = json.loads(raw)
            accepted.append(sample)
            accepted_parent_ids.add(str(sample["quality"]["parent_sample_id"]))
    pending = [sample for sample in parents if str(sample["sample_id"]) not in accepted_parent_ids]
    rejected: Counter[str] = Counter()
    attempted = 0
    last_wave_started = 0.0

    def request_group(
        group: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, str] | None, str | None]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=grouped_rewrite_messages(group),
                response_format={"type": "json_object"},
                temperature=0.6 if provider == "kimi" else 0.3,
                extra_body={"thinking": {"type": "disabled"}} if provider == "kimi" else {},
                timeout=300,
            )
            content = response.choices[0].message.content or ""
            questions = _extract_grouped_questions(
                content, {str(sample["sample_id"]) for sample in group}
            )
            return group, questions, None
        except Exception as exc:
            return group, None, type(exc).__name__

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _attempt in range(max_attempts):
            if not pending:
                break
            next_pending: list[dict[str, Any]] = []
            groups = [
                pending[offset : offset + group_size]
                for offset in range(0, len(pending), group_size)
            ]
            for wave_offset in range(0, len(groups), max_workers):
                wave = groups[wave_offset : wave_offset + max_workers]
                wait_for = min_interval_seconds - (time.monotonic() - last_wave_started)
                if wait_for > 0:
                    time.sleep(wait_for)
                last_wave_started = time.monotonic()
                attempted += len(wave)
                outcomes = executor.map(request_group, wave)
                for group, questions, error_name in outcomes:
                    if questions is None:
                        rejected[error_name or "unknown_error"] += len(group)
                        next_pending.extend(group)
                        continue
                    newly_accepted: list[dict[str, Any]] = []
                    for parent in group:
                        question = questions[str(parent["sample_id"])]
                        reason = validate_rewrite(
                            str(parent["question"]), question, str(parent["language"])
                        )
                        if reason is not None:
                            rejected[reason] += 1
                            next_pending.append(parent)
                            continue
                        sample = make_rewrite(parent, question, provider, model)
                        newly_accepted.append(sample)
                        accepted.append(sample)
                        accepted_parent_ids.add(str(parent["sample_id"]))
                    if newly_accepted:
                        with destination.open("a", encoding="utf-8") as handle:
                            for sample in newly_accepted:
                                handle.write(
                                    json.dumps(
                                        sample,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )
                            handle.flush()
                    print(
                        f"API rewrite progress: {len(accepted)}/{len(parents)} accepted; "
                        f"request {attempted}",
                        flush=True,
                    )
            pending = next_pending
    if pending:
        rejected["exhausted_parent"] += len(pending)
    if accepted:
        validate_jsonl(destination, check_assets=True)
    return RewriteResult(
        accepted=len(accepted),
        attempted=attempted,
        rejected=dict(sorted(rejected.items())),
        provider=provider,
        model=model,
        output=str(destination),
    )


def generate_rewrites_batch(
    parents: Iterable[dict[str, Any]],
    *,
    keys: APIKeys,
    destination: Path,
    work_dir: Path,
    choice: int,
    noul: int,
    provider: str = "kimi",
    poll_seconds: int = 30,
) -> BatchRewriteResult:
    """Submit or resume an OpenAI-compatible Batch job and build exact task quotas."""
    parent_rows = list(parents)
    client, model = _client(provider, keys)
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "requests.jsonl"
    parent_path = work_dir / "parents.jsonl"
    state_path = work_dir / "batch-state.json"
    raw_output_path = work_dir / "responses.jsonl"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        batch_id = str(state["batch_id"])
    else:
        with (
            request_path.open("w", encoding="utf-8") as requests,
            parent_path.open("w", encoding="utf-8") as parent_file,
        ):
            for index, parent in enumerate(parent_rows):
                custom_id = f"rewrite-{index:06d}"
                body: dict[str, Any] = {
                    "model": model,
                    "messages": rewrite_messages(parent),
                    "response_format": {"type": "json_object"},
                }
                if provider == "glm":
                    body["temperature"] = 0.3
                if provider == "kimi":
                    body["thinking"] = {"type": "disabled"}
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                requests.write(json.dumps(request, ensure_ascii=False) + "\n")
                parent_file.write(
                    json.dumps({"custom_id": custom_id, "sample": parent}, ensure_ascii=False)
                    + "\n"
                )
        with request_path.open("rb") as handle:
            upload = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="1d",
        )
        batch_id = str(batch.id)
        state_path.write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "input_file_id": str(upload.id),
                    "provider": provider,
                    "model": model,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        status = str(batch.status)
        counts = getattr(batch, "request_counts", None)
        print(f"API batch {batch_id}: {status} {counts or ''}", flush=True)
        if status in terminal:
            break
        time.sleep(poll_seconds)
    if status != "completed" or not batch.output_file_id:
        raise RuntimeError(f"API batch {batch_id} ended with status {status}")
    content = client.files.content(batch.output_file_id)
    content.write_to_file(raw_output_path)

    parent_by_id = {
        row["custom_id"]: row["sample"]
        for row in (
            json.loads(raw) for raw in parent_path.read_text(encoding="utf-8").splitlines() if raw
        )
    }
    accepted_by_task: dict[str, list[dict[str, Any]]] = {"choice": [], "noul": []}
    rejected: Counter[str] = Counter()
    for raw in raw_output_path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        result = json.loads(raw)
        batch_parent = parent_by_id.get(result.get("custom_id"))
        if batch_parent is None:
            rejected["unknown_custom_id"] += 1
            continue
        response = result.get("response", {})
        if response.get("status_code") != 200:
            rejected[f"http_{response.get('status_code', 'unknown')}"] += 1
            continue
        try:
            message = response["body"]["choices"][0]["message"]
            question = _extract_question(message.get("content") or "")
            reason = validate_rewrite(
                str(batch_parent["question"]), question, str(batch_parent["language"])
            )
            if reason is not None:
                rejected[reason] += 1
                continue
            task = str(batch_parent["task_type"])
            accepted_by_task[task].append(make_rewrite(batch_parent, question, provider, model))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rejected[type(exc).__name__] += 1

    targets = {"choice": choice, "noul": noul}
    selected: list[dict[str, Any]] = []
    for task, target in targets.items():
        if len(accepted_by_task[task]) < target:
            raise ValueError(
                f"batch produced only {len(accepted_by_task[task])}/{target} valid {task} rows"
            )
        selected.extend(accepted_by_task[task][:target])
    selected.sort(key=lambda sample: str(sample["sample_id"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in selected:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    validate_jsonl(destination, check_assets=True)
    return BatchRewriteResult(
        accepted=len(selected),
        requested=len(parent_rows),
        rejected=dict(sorted(rejected.items())),
        task_counts=targets,
        provider=provider,
        model=model,
        batch_id=batch_id,
        output=str(destination),
    )
