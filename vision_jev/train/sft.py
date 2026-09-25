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
from torch.utils.data import DataLoader, Dataset
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
            correct = parsed == json.loads(expected)
            counts["syntax_valid"] += int(syntax_valid)
            counts["correct"] += int(correct)
            counts[f"{task}:correct"] += int(correct)
            source_counts[f"{source}:correct"] += int(correct)
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


def train_sft(
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    *,
    model_root: Path = DEFAULT_MODEL_CACHE,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="bf16" if config.get("bf16", True) else "no",
    )
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
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["microbatch_questions_per_device"]),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(config.get("dataloader_workers", 0)),
        generator=generator,
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
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    step = 0
    first_loss: float | None = None
    started = time.monotonic()
    model.train()
    for epoch in range(int(config["epochs"])):
        for batch in train_loader:
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
                if accelerator.is_main_process and (
                    step == 1 or step % int(config.get("log_every", 10)) == 0
                ):
                    record = {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": reduced_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                if step >= total_steps:
                    break
        if step >= total_steps:
            break
    eval_loss = _evaluate(model, eval_loader, accelerator, int(config.get("eval_batches", 50)))
    accelerator.wait_for_everyone()
    final_loss = reduced_loss
    summary = {
        "steps": step,
        "train_questions": len(train_data),
        "eval_questions": len(eval_data),
        "first_train_loss": first_loss,
        "final_train_loss": final_loss,
        "eval_loss": eval_loss,
        "elapsed_seconds": time.monotonic() - started,
        "world_size": accelerator.num_processes,
    }
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir / "checkpoint-last", safe_serialization=True)
        processor.save_pretrained(output_dir / "checkpoint-last")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    return summary
