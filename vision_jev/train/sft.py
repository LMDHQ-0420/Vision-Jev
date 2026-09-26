"""Answer-only multimodal SFT using the native Qwen3.5 processor."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import get_linear_schedule_with_warmup

from vision_jev.model.qwen35 import DEFAULT_MODEL_CACHE, load_backbone, load_processor

SYSTEM_PROMPT = (
    "You are Vision-Jev. Read the visual evidence and return only the requested compact JSON."
)

SCORE_LEVELS = {
    1: "very poor: lowest quality, with severe visible degradation",
    2: "poor: below-average quality, with clear degradation",
    3: "fair: middle quality, with noticeable but limited defects",
    4: "good: above-average quality, with only minor defects",
    5: "excellent: highest quality, clean and visually pleasing",
}


def validate_training_config(config: dict[str, Any], world_size: int) -> None:
    required = (
        "epochs",
        "microbatch_questions_per_device",
        "gradient_accumulation_steps",
        "learning_rate",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"training config is missing required keys: {missing}")
    if int(config["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    microbatch = int(config["microbatch_questions_per_device"])
    accumulation = int(config["gradient_accumulation_steps"])
    if microbatch < 1 or accumulation < 1:
        raise ValueError("microbatch and gradient accumulation must be positive")
    expected_world_size = int(config.get("expected_world_size", world_size))
    if expected_world_size != world_size:
        raise ValueError(
            f"config expects world_size={expected_world_size}, "
            f"but launch has world_size={world_size}"
        )
    actual_global_batch = microbatch * accumulation * world_size
    configured_global_batch = int(config.get("global_batch_questions", actual_global_batch))
    if configured_global_batch != actual_global_batch:
        raise ValueError(
            f"global batch mismatch: configured {configured_global_batch}, "
            f"derived {actual_global_batch}"
        )
    checkpoint_every = int(config.get("checkpoint_every_steps", 0))
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every_steps cannot be negative")


class EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch ordering that can be reconstructed during resume."""

    def __init__(self, data_source: ManifestDataset, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Any:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def answer_text(sample: dict[str, Any]) -> str:
    task = str(sample["task_type"])
    value: bool | int | str
    if task == "noul":
        value = bool(sample["target"])
    elif task == "score":
        value = int(str(sample["target"]).rsplit("_", 1)[-1])
    else:
        target = sample["target"]
        value = str(target[0] if isinstance(target, list) else target)
    return json.dumps({task: value}, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=4096)
def _image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _normalized_box(sample: dict[str, Any], box: list[float]) -> list[int]:
    image = sample.get("image")
    if not image:
        return [round(value) for value in box]
    width, height = _image_size(str(image))
    scales = (width, height, width, height)
    return [
        max(0, min(1000, round(float(value) / scale * 1000)))
        for value, scale in zip(box, scales, strict=True)
    ]


def _render_options(sample: dict[str, Any]) -> str:
    rendered: list[str] = []
    for item in sample.get("options", []):
        line = f"- {item['id']}: {item['text']}"
        if "box" in item:
            coordinates = ",".join(str(value) for value in _normalized_box(sample, item["box"]))
            line += f"; box_0_1000=[{coordinates}]"
        rendered.append(line)
    return "\n".join(rendered)


def question_text(sample: dict[str, Any]) -> str:
    sections = [f"Task: {sample['task_type']}"]
    state = str(sample.get("state_text", "")).strip()
    if state:
        sections.append(f"Visible state: {state}")
    history = sample.get("allowed_history", [])
    if history:
        rendered_history = "\n".join(
            f"{index}. {json.dumps(item, ensure_ascii=False)}"
            for index, item in enumerate(history, 1)
        )
        sections.append(f"Completed steps (oldest to newest):\n{rendered_history}")
    sections.append(f"Question: {sample['question']}")
    options = sample.get("options", [])
    if options:
        sections.append(f"Candidates:\n{_render_options(sample)}")
    if any("box" in item for item in options):
        sections.append(
            "Box coordinates are normalized to 0-1000 as [x1,y1,x2,y2]. "
            "Match the referring expression against both the image and each candidate box."
        )
    if sample["task_type"] == "noul":
        sections.append('Return exactly {"noul":true} or {"noul":false}.')
    elif sample["task_type"] == "score":
        levels = "\n".join(f"{level} = {meaning}" for level, meaning in SCORE_LEVELS.items())
        sections.append(
            "Judge perceptual image quality from blur, noise, exposure, color, compression, "
            f"and overall appearance. The ordered levels are:\n{levels}"
        )
        sections.append('Return exactly {"score":<integer from 1 to 5>}.')
    else:
        sections.append(f'Return exactly {{"{sample["task_type"]}":"<candidate_id>"}}.')
    return "\n\n".join(sections)


def conversation(sample: dict[str, Any], *, include_answer: bool) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    image = sample.get("image")
    if image:
        content.append({"type": "image", "image": str(image)})
    content.append({"type": "text", "text": question_text(sample)})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer_text(sample)}]}
        )
    return messages


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        path: Path,
        role: str,
        maximum: int | None = None,
        seed: int = 42,
        task_repeat: dict[str, int] | None = None,
        include_sources: list[str] | None = None,
    ) -> None:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("pilot_role", "train") == role and (
                    include_sources is None or str(row["source"]) in include_sources
                ):
                    rows.append(row)
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}\0{row['sample_id']}".encode()).digest())
        rows = rows[:maximum] if maximum is not None else rows
        repeat = task_repeat or {}
        self.rows = [
            row for row in rows for _ in range(max(1, int(repeat.get(str(row["task_type"]), 1))))
        ]
        if not self.rows:
            raise ValueError(f"manifest contains no {role!r} rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass
class NativeQwenCollator:
    processor: Any
    image_token_target: int = 256
    region_image_token_target: int = 576
    score_image_token_target: int = 576

    def visual_budget(self, samples: list[dict[str, Any]]) -> int:
        budget = self.image_token_target
        if any(sample["task_type"] == "score" for sample in samples):
            budget = max(budget, self.score_image_token_target)
        if any(any("box" in option for option in sample.get("options", [])) for sample in samples):
            budget = max(budget, self.region_image_token_target)
        return budget

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full = [conversation(sample, include_answer=True) for sample in samples]
        prompts = [conversation(sample, include_answer=False) for sample in samples]
        processor_kwargs = {
            "padding": True,
            "images_kwargs": {
                "size": {
                    "shortest_edge": 65_536,
                    "longest_edge": self.visual_budget(samples) * 16 * 16 * 4,
                }
            },
        }
        batch = self.processor.apply_chat_template(
            full,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        prompt_batch = self.processor.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for index, length in enumerate(prompt_lengths):
            prompt_ids = prompt_batch["input_ids"][index, :length]
            if not torch.equal(batch["input_ids"][index, :length], prompt_ids):
                raise ValueError(f"chat-template prefix mismatch for {samples[index]['sample_id']}")
            labels[index, :length] = -100
        batch["labels"] = labels
        return dict(batch)


def evaluate_checkpoint(
    config_path: Path,
    data_path: Path,
    checkpoint: Path,
    output_path: Path,
    *,
    model_root: Path = DEFAULT_MODEL_CACHE,
    maximum: int | None = 300,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Greedy exact-match evaluation for the structured SFT response."""
    from collections import Counter

    from peft import PeftModel

    if maximum is not None and maximum < 1:
        raise ValueError("maximum must be positive or None")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    processor = load_processor(Path(config["model_config"]), model_root)
    base = load_backbone(
        Path(config["model_config"]),
        model_root=model_root,
        dtype=torch.bfloat16,
        use_lora=False,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("SFT generation evaluation requires a CUDA device")
    model = PeftModel.from_pretrained(base, checkpoint).to("cuda").eval()
    dataset = ManifestDataset(data_path, "eval", maximum, seed=int(config.get("seed", 42)))
    collator = NativeQwenCollator(
        processor,
        image_token_target=int(config.get("image_token_target", 256)),
        region_image_token_target=int(config.get("region_image_token_target", 576)),
        score_image_token_target=int(config.get("score_image_token_target", 576)),
    )
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    score_confusion: Counter[tuple[int, int]] = Counter()
    score_absolute_error = 0
    score_within_one = 0
    latencies: list[float] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output, torch.no_grad():
        for sample_index in range(len(dataset)):
            sample = dataset[sample_index]
            batch = processor.apply_chat_template(
                conversation(sample, include_answer=False),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "images_kwargs": {
                        "size": {
                            "shortest_edge": 65_536,
                            "longest_edge": collator.visual_budget([sample]) * 16 * 16 * 4,
                        }
                    }
                },
            ).to("cuda")
            input_length = int(batch["input_ids"].shape[1])
            generation_started = time.monotonic()
            generated = model.generate(  # type: ignore[no-untyped-call]
                **batch,
                max_new_tokens=24,
                do_sample=False,
                use_cache=True,
            )
            latencies.append(time.monotonic() - generation_started)
            prediction = processor.decode(
                generated[0, input_length:], skip_special_tokens=True
            ).strip()
            expected = answer_text(sample)
            expected_object = json.loads(expected)
            task = str(sample["task_type"])
            source = str(sample["source"])
            counts["questions"] += 1
            counts[f"{task}:questions"] += 1
            source_counts[f"{source}:questions"] += 1
            try:
                parsed = json.loads(prediction)
                syntax_valid = isinstance(parsed, dict)
            except json.JSONDecodeError:
                parsed = None
                syntax_valid = False
            correct = parsed == expected_object
            counts["syntax_valid"] += int(syntax_valid)
            counts["correct"] += int(correct)
            counts[f"{task}:correct"] += int(correct)
            source_counts[f"{source}:correct"] += int(correct)
            if task == "score" and isinstance(parsed, dict):
                expected_score = expected_object.get("score")
                predicted_score = parsed.get("score")
                if (
                    isinstance(expected_score, int)
                    and not isinstance(expected_score, bool)
                    and isinstance(predicted_score, int)
                    and not isinstance(predicted_score, bool)
                    and 1 <= predicted_score <= 5
                ):
                    absolute_error = abs(expected_score - predicted_score)
                    counts["score:ordinal_valid"] += 1
                    score_absolute_error += absolute_error
                    score_within_one += int(absolute_error <= 1)
                    score_confusion[(expected_score, predicted_score)] += 1
            output.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "task_type": task,
                        "expected": expected,
                        "prediction": prediction,
                        "correct": correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (sample_index + 1) % progress_every == 0 or sample_index + 1 == len(dataset):
                output.flush()
                print(
                    json.dumps(
                        {
                            "evaluated": sample_index + 1,
                            "total": len(dataset),
                            "exact_match": counts["correct"] / (sample_index + 1),
                            "syntax_valid_rate": counts["syntax_valid"] / (sample_index + 1),
                        }
                    ),
                    flush=True,
                )
    questions = counts["questions"]
    if len(latencies) == 1:
        p50 = p95 = latencies[0]
    else:
        percentiles = statistics.quantiles(latencies, n=100, method="inclusive")
        p50, p95 = percentiles[49], percentiles[94]
    report: dict[str, Any] = {
        "questions": questions,
        "syntax_valid_rate": counts["syntax_valid"] / questions,
        "exact_match": counts["correct"] / questions,
        "by_task": {},
        "by_source": {},
        "elapsed_seconds": time.monotonic() - started,
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": p50,
            "p95": p95,
        },
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    for task in ("choice", "noul", "score"):
        task_questions = counts[f"{task}:questions"]
        if task_questions:
            report["by_task"][task] = {
                "questions": task_questions,
                "exact_match": counts[f"{task}:correct"] / task_questions,
            }
            if task == "score":
                ordinal_valid = counts["score:ordinal_valid"]
                report["by_task"][task]["ordinal_valid_rate"] = ordinal_valid / task_questions
                if ordinal_valid:
                    report["by_task"][task]["mean_absolute_error"] = (
                        score_absolute_error / ordinal_valid
                    )
                    report["by_task"][task]["within_one_accuracy"] = (
                        score_within_one / ordinal_valid
                    )
                    report["by_task"][task]["confusion"] = {
                        f"{expected}->{predicted}": count
                        for (expected, predicted), count in sorted(score_confusion.items())
                    }
    sources = sorted({key.rsplit(":", 1)[0] for key in source_counts})
    for source in sources:
        source_questions = source_counts[f"{source}:questions"]
        report["by_source"][source] = {
            "questions": source_questions,
            "exact_match": source_counts[f"{source}:correct"] / source_questions,
        }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _evaluate(model: Any, loader: DataLoader[Any], accelerator: Accelerator, limit: int) -> float:
    model.eval()
    total = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            loss = model(**batch).loss.detach().double()
            total[0] += loss
            total[1] += 1
    total = accelerator.reduce(total, reduction="sum")
    model.train()
    return float((total[0] / total[1].clamp_min(1)).item())


def _save_training_checkpoint(
    destination: Path,
    *,
    accelerator: Accelerator,
    model: Any,
    processor: Any,
    optimizer: Any,
    trainer_state: dict[str, Any],
) -> None:
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=False)
        accelerator.unwrap_model(model).save_pretrained(destination, safe_serialization=True)
        processor.save_pretrained(destination)
        (destination / "trainer-state.json").write_text(
            json.dumps(trainer_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    accelerator.save_state(str(destination / "accelerate-state"), safe_serialization=True)
    unwrapped = accelerator.unwrap_model(model)
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    names = {id(parameter): name for name, parameter in unwrapped.named_parameters()}
    named_state = {
        names[id(parameter)]: state
        for parameter, state in raw_optimizer.state.items()
        if id(parameter) in names
    }
    named_groups = []
    for group in raw_optimizer.param_groups:
        named_groups.append(
            {
                **{key: value for key, value in group.items() if key != "params"},
                "param_names": [names[id(parameter)] for parameter in group["params"]],
            }
        )
    torch.save(
        {"state": named_state, "param_groups": named_groups},
        destination / f"optimizer-named-rank-{accelerator.process_index}.pt",
    )
    accelerator.wait_for_everyone()


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _load_named_optimizer_state(
    checkpoint: Path, *, accelerator: Accelerator, model: Any, optimizer: Any
) -> None:
    state_path = checkpoint / f"optimizer-named-rank-{accelerator.process_index}.pt"
    saved = torch.load(state_path, map_location="cpu", weights_only=False)
    unwrapped = accelerator.unwrap_model(model)
    parameters = dict(unwrapped.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    raw_optimizer.state.clear()
    for name, state in saved["state"].items():
        parameter = parameters[name]
        raw_optimizer.state[parameter] = _to_device(state, parameter.device)
    if len(raw_optimizer.param_groups) != len(saved["param_groups"]):
        raise ValueError("optimizer parameter-group count differs from the checkpoint")
    for current, previous in zip(
        raw_optimizer.param_groups, saved["param_groups"], strict=True
    ):
        current_names = [names[id(parameter)] for parameter in current["params"]]
        if set(current_names) != set(previous["param_names"]):
            raise ValueError("optimizer trainable parameter names differ from the checkpoint")
        for key, value in previous.items():
            if key != "param_names":
                current[key] = value


def train_sft(
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    *,
    model_root: Path = DEFAULT_MODEL_CACHE,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="bf16" if config.get("bf16", True) else "no",
    )
    validate_training_config(config, accelerator.num_processes)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"training output already exists and is not empty: {output_dir}")
    if resume_from is not None and config.get("initial_adapter"):
        raise ValueError("resume_from and initial_adapter cannot be used together")
    processor = load_processor(Path(config["model_config"]), model_root)
    initial_adapter = config.get("initial_adapter")
    if initial_adapter:
        from peft import PeftModel

        base = load_backbone(
            Path(config["model_config"]),
            model_root=model_root,
            dtype=torch.bfloat16 if config.get("bf16", True) else torch.float32,
            use_lora=False,
        )
        model = PeftModel.from_pretrained(base, Path(initial_adapter), is_trainable=True)
    else:
        model = load_backbone(
            Path(config["model_config"]),
            model_root=model_root,
            dtype=torch.bfloat16 if config.get("bf16", True) else torch.float32,
            use_lora=True,
            lora_rank=int(config.get("lora_rank", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
        )
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    train_data = ManifestDataset(
        data_path,
        "train",
        config.get("max_train_questions"),
        seed=seed,
        task_repeat=config.get("task_repeat"),
        include_sources=config.get("include_sources"),
    )
    eval_data = ManifestDataset(data_path, "eval", config.get("max_eval_questions"), seed=seed)
    collator = NativeQwenCollator(
        processor,
        image_token_target=int(config.get("image_token_target", 256)),
        region_image_token_target=int(config.get("region_image_token_target", 576)),
        score_image_token_target=int(config.get("score_image_token_target", 576)),
    )
    train_sampler = EpochRandomSampler(train_data, seed)
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["microbatch_questions_per_device"]),
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=int(config.get("dataloader_workers", 0)),
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=int(config["microbatch_questions_per_device"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(config.get("dataloader_workers", 0)),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    # Accelerate advances a wrapped scheduler once per process for sharded batches.
    # The pre-sharding count therefore keeps the global LR curve aligned with all epochs.
    updates_per_epoch = math.ceil(len(train_loader) / int(config["gradient_accumulation_steps"]))
    total_steps = int(config.get("max_steps") or updates_per_epoch * int(config["epochs"]))
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer,
        num_warmup_steps=max(1, int(total_steps * float(config.get("warmup_ratio", 0.03)))),
        num_training_steps=total_steps,
    )
    resume_trainer_state: dict[str, Any] | None = None
    if resume_from is not None:
        resume_trainer_state = json.loads(
            (resume_from / "trainer-state.json").read_text(encoding="utf-8")
        )
        if int(resume_trainer_state["world_size"]) != accelerator.num_processes:
            raise ValueError("resume checkpoint world size differs from the current launch")
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )
    if resume_from is not None:
        accelerator.load_state(str(resume_from / "accelerate-state"))
        _load_named_optimizer_state(
            resume_from, accelerator=accelerator, model=model, optimizer=optimizer
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    step = int(resume_trainer_state["step"]) if resume_trainer_state else 0
    first_loss = resume_trainer_state.get("first_train_loss") if resume_trainer_state else None
    start_epoch = int(resume_trainer_state["epoch"]) if resume_trainer_state else 0
    resume_batches = (
        int(resume_trainer_state["batches_in_epoch"]) if resume_trainer_state else 0
    )
    elapsed_offset = (
        float(resume_trainer_state.get("elapsed_seconds", 0.0)) if resume_trainer_state else 0.0
    )
    started = time.monotonic()
    last_loss = resume_trainer_state.get("last_train_loss") if resume_trainer_state else None
    checkpoint_every = int(config.get("checkpoint_every_steps", 0))
    model.train()
    for epoch in range(start_epoch, int(config["epochs"])):
        train_sampler.set_epoch(epoch)
        skipped_batches = resume_batches if epoch == start_epoch else 0
        epoch_loader = (
            accelerator.skip_first_batches(train_loader, skipped_batches)
            if skipped_batches
            else train_loader
        )
        for batch_index, batch in enumerate(epoch_loader, start=skipped_batches):
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), float(config.get("gradient_clip", 1.0))
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean").float().item()
                if first_loss is None:
                    first_loss = reduced_loss
                last_loss = reduced_loss
                if accelerator.is_main_process and (
                    step == 1 or step % int(config.get("log_every", 10)) == 0
                ):
                    record = {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": reduced_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "elapsed_seconds": elapsed_offset + time.monotonic() - started,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                if checkpoint_every and step % checkpoint_every == 0:
                    _save_training_checkpoint(
                        output_dir / f"checkpoint-step-{step:06d}",
                        accelerator=accelerator,
                        model=model,
                        processor=processor,
                        optimizer=optimizer,
                        trainer_state={
                            "schema_version": 1,
                            "step": step,
                            "epoch": epoch,
                            "batches_in_epoch": batch_index + 1,
                            "world_size": accelerator.num_processes,
                            "first_train_loss": first_loss,
                            "last_train_loss": last_loss,
                            "elapsed_seconds": elapsed_offset + time.monotonic() - started,
                        },
                    )
                if step >= total_steps:
                    break
        resume_batches = 0
        if step >= total_steps:
            break
    eval_loss = _evaluate(model, eval_loader, accelerator, int(config.get("eval_batches", 50)))
    accelerator.wait_for_everyone()
    if last_loss is None:
        raise RuntimeError("training completed without an optimizer step")
    summary = {
        "steps": step,
        "train_questions": len(train_data),
        "eval_questions": len(eval_data),
        "first_train_loss": first_loss,
        "final_train_loss": last_loss,
        "eval_loss": eval_loss,
        "elapsed_seconds": elapsed_offset + time.monotonic() - started,
        "world_size": accelerator.num_processes,
        "global_batch_questions": (
            int(config["microbatch_questions_per_device"])
            * int(config["gradient_accumulation_steps"])
            * accelerator.num_processes
        ),
    }
    _save_training_checkpoint(
        output_dir / "checkpoint-last",
        accelerator=accelerator,
        model=model,
        processor=processor,
        optimizer=optimizer,
        trainer_state={
            "schema_version": 1,
            "step": step,
            "epoch": int(config["epochs"]),
            "batches_in_epoch": 0,
            "world_size": accelerator.num_processes,
            "first_train_loss": first_loss,
            "last_train_loss": last_loss,
            "elapsed_seconds": summary["elapsed_seconds"],
        },
    )
    if accelerator.is_main_process:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    return summary
