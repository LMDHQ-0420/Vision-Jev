"""Static RLCD-inspired training for calibrated native decision heads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator  # type: ignore[import-untyped]
from accelerate.utils import DistributedDataParallelKwargs  # type: ignore[import-untyped]
from peft import PeftModel
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from vision_jev.model.heads import ChoiceHead, NoulHead, ScoreHead
from vision_jev.model.losses import choice_set_loss, ranked_probability_score
from vision_jev.model.qwen35 import DEFAULT_MODEL_CACHE, load_backbone, load_processor
from vision_jev.train.sft import NativeQwenCollator, _set_process_name, conversation

TASKS = ("choice", "noul", "score")


class RlcdDataset(Dataset[dict[str, Any]]):
    """Deterministically ordered rows selected by the immutable decision role."""

    def __init__(
        self,
        path: Path,
        role: str,
        *,
        seed: int,
        maximum: int | None = None,
    ) -> None:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    row = json.loads(raw)
                    if row.get("decision_role") == role:
                        rows.append(row)
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row['sample_id']}".encode()
            ).digest()
        )
        self.rows = rows[:maximum] if maximum is not None else rows
        if not self.rows:
            raise ValueError(f"manifest contains no RLCD rows for role={role!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _one_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError("RLCD currently requires a per-device microbatch of one")
    return rows[0]


class DecisionHeads(nn.Module):
    def __init__(self, hidden_size: int, width: int, attention_heads: int) -> None:
        super().__init__()
        self.choice = ChoiceHead(hidden_size, width, attention_heads)
        self.noul = NoulHead(hidden_size, width)
        self.score = ScoreHead(hidden_size, width)

    def forward(
        self,
        task: str,
        question: Tensor,
        candidates: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if task == "noul":
            return self.noul(question)
        if candidates is None or mask is None:
            raise ValueError(f"{task} requires candidate features and a mask")
        if task == "choice":
            return self.choice(question, candidates, mask)
        if task == "score":
            return self.score(question, candidates, mask)
        raise ValueError(f"unsupported task: {task}")


def validate_rlcd_config(config: dict[str, Any], world_size: int) -> None:
    required = {
        "epochs",
        "microbatch_questions_per_device",
        "gradient_accumulation_steps",
        "learning_rate",
        "sft_checkpoint",
        "train_manifest",
        "role_manifest",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"RLCD config is missing required keys: {missing}")
    microbatch = int(config["microbatch_questions_per_device"])
    accumulation = int(config["gradient_accumulation_steps"])
    if microbatch != 1:
        raise ValueError("RLCD microbatch_questions_per_device must be one")
    expected_world_size = int(config.get("expected_world_size", world_size))
    if expected_world_size != world_size:
        raise ValueError(
            f"config expects world_size={expected_world_size}, "
            f"but launch has world_size={world_size}"
        )
    expected_batch = microbatch * accumulation * world_size
    if int(config.get("global_batch_questions", expected_batch)) != expected_batch:
        raise ValueError("configured RLCD global batch does not match the distributed launch")


class FeatureExtractor:
    """Frozen SFT prompt and candidate features consumed by native heads."""

    def __init__(self, model: Any, processor: Any, config: dict[str, Any], device: Any) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.budget = NativeQwenCollator(
            processor,
            image_token_target=int(config.get("image_token_target", 256)),
            region_image_token_target=int(config.get("region_image_token_target", 576)),
            score_image_token_target=int(config.get("score_image_token_target", 576)),
        )

    def _candidate_features(self, sample: dict[str, Any]) -> Tensor | None:
        options = sample.get("options", [])
        if not options:
            return None
        texts = [f"{item['id']}: {item['text']}" for item in options]
        encoded = self.processor.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        embeddings = self.model.get_input_embeddings()(encoded["input_ids"])
        weights = encoded["attention_mask"].to(embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return pooled.unsqueeze(0)

    def __call__(self, sample: dict[str, Any]) -> tuple[Tensor, Tensor | None, Tensor | None]:
        prompt = self.processor.apply_chat_template(
            conversation(sample, include_answer=False),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "images_kwargs": {
                    "size": {
                        "shortest_edge": 65_536,
                        "longest_edge": self.budget.visual_budget([sample]) * 16 * 16 * 4,
                    }
                }
            },
        ).to(self.device)
        backbone = self.model.base_model.model.model
        output = backbone(**prompt, return_dict=True, use_cache=False)
        length = int(prompt["attention_mask"].sum().item())
        question = output.last_hidden_state[:, length - 1]
        candidates = self._candidate_features(sample)
        mask = None
        if candidates is not None:
            mask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=self.device)
        return question, candidates, mask


def _target_indices(sample: dict[str, Any]) -> list[int]:
    target = sample["target"]
    targets = target if isinstance(target, list) else [target]
    index = {str(option["id"]): position for position, option in enumerate(sample["options"])}
    try:
        return [index[str(value)] for value in targets]
    except KeyError as exc:
        raise ValueError(f"target is absent from options for {sample['sample_id']}") from exc


def decision_loss(task: str, logits: Tensor, probs: Tensor, sample: dict[str, Any]) -> Tensor:
    if task == "noul":
        target = torch.tensor([float(bool(sample["target"]))], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits.float(), target)
    indices = _target_indices(sample)
    if task == "choice":
        valid = torch.zeros_like(probs)
        valid[0, indices] = 1
        return choice_set_loss(probs, valid)
    if task == "score":
        target = torch.tensor([indices[0]], device=logits.device)
        mask = torch.ones_like(probs, dtype=torch.bool)
        nll = F.nll_loss(probs.float().clamp_min(1e-12).log(), target)
        return nll + 0.5 * ranked_probability_score(probs, target, mask)
    raise ValueError(f"unsupported task: {task}")


def _prediction(
    task: str, logits: Tensor, probs: Tensor, sample: dict[str, Any]
) -> dict[str, float]:
    if task == "noul":
        probability = float(probs.item())
        target = int(bool(sample["target"]))
        predicted = int(probability >= 0.5)
        confidence = max(probability, 1.0 - probability)
        nll = -math.log(max(probability if target else 1.0 - probability, 1e-12))
        brier = 2.0 * (probability - target) ** 2
        return {
            "correct": float(predicted == target),
            "confidence": confidence,
            "nll": nll,
            "brier": brier,
            "rps": 0.0,
            "absolute_error": 0.0,
        }
    values = probs[0].float().cpu()
    targets = _target_indices(sample)
    predicted = int(values.argmax().item())
    target_distribution = torch.zeros_like(values)
    target_distribution[targets] = 1.0 / len(targets)
    target_mass = float(values[targets].sum().item())
    result = {
        "correct": float(predicted in targets),
        "confidence": float(values.max().item()),
        "nll": -math.log(max(target_mass, 1e-12)),
        "brier": float((values - target_distribution).square().sum().item()),
        "rps": 0.0,
        "absolute_error": 0.0,
    }
    if task == "score":
        result["rps"] = float(
            (values.cumsum(0)[:-1] - target_distribution.cumsum(0)[:-1]).square().mean().item()
        )
        result["absolute_error"] = float(abs(predicted - targets[0]))
    return result


def _new_totals(device: Any) -> Tensor:
    # questions, correct, nll, brier, rps, absolute_error, plus 15-bin ECE triples
    return torch.zeros((len(TASKS), 6 + 15 * 3), dtype=torch.float64, device=device)


def _record(totals: Tensor, task: str, prediction: dict[str, float]) -> None:
    row = TASKS.index(task)
    totals[row, 0] += 1
    for offset, key in enumerate(("correct", "nll", "brier", "rps", "absolute_error"), 1):
        totals[row, offset] += prediction[key]
    bucket = min(14, int(prediction["confidence"] * 15))
    base = 6 + bucket * 3
    totals[row, base] += 1
    totals[row, base + 1] += prediction["confidence"]
    totals[row, base + 2] += prediction["correct"]


def _report(totals: Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {"questions": int(totals[:, 0].sum().item()), "by_task": {}}
    accuracies = []
    for row, task in enumerate(TASKS):
        questions = float(totals[row, 0].item())
        if not questions:
            continue
        ece = 0.0
        for bucket in range(15):
            base = 6 + bucket * 3
            count = float(totals[row, base].item())
            if count:
                confidence = float(totals[row, base + 1].item()) / count
                accuracy = float(totals[row, base + 2].item()) / count
                ece += count / questions * abs(confidence - accuracy)
        metrics = {
            "questions": int(questions),
            "accuracy": float(totals[row, 1].item()) / questions,
            "nll": float(totals[row, 2].item()) / questions,
            "brier": float(totals[row, 3].item()) / questions,
            "ece_15": ece,
        }
        if task == "score":
            metrics["rps"] = float(totals[row, 4].item()) / questions
            metrics["mean_absolute_error"] = float(totals[row, 5].item()) / questions
        result["by_task"][task] = metrics
        accuracies.append(metrics["accuracy"])
    result["macro_accuracy"] = sum(accuracies) / len(accuracies)
    return result


def evaluate_heads(
    heads: Any,
    extractor: FeatureExtractor,
    loader: DataLoader[Any],
    accelerator: Accelerator,
) -> dict[str, Any]:
    heads.eval()
    totals = _new_totals(accelerator.device)
    with torch.no_grad():
        for sample in loader:
            task = str(sample["task_type"])
            with accelerator.autocast():
                question, candidates, mask = extractor(sample)
                logits, probs = heads(task, question, candidates, mask)
            _record(totals, task, _prediction(task, logits, probs, sample))
    totals = accelerator.reduce(totals, reduction="sum")
    heads.train()
    return _report(totals.cpu())


def _save_heads(
    destination: Path,
    *,
    accelerator: Accelerator,
    heads: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=False)
        tensors = {
            name: value.detach().cpu().contiguous()
            for name, value in accelerator.unwrap_model(heads).state_dict().items()
        }
        save_file(tensors, destination / "decision_heads.safetensors")
        (destination / "trainer-state.json").write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        provenance = {
            "stage": "rlcd_static",
            "sft_checkpoint": config["sft_checkpoint"],
            "model_config": config["model_config"],
            "head_width": config["head_width"],
            "head_attention_heads": config["head_attention_heads"],
        }
        (destination / "rlcd_config.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()


def _temperature_grid(
    logits: list[list[float]], targets: list[list[int]], config: dict[str, Any]
) -> float:
    if not logits:
        return 1.0
    low = math.log(float(config["temperature_grid_min"]))
    high = math.log(float(config["temperature_grid_max"]))
    steps = int(config["temperature_grid_steps"])
    best = (float("inf"), 1.0)
    for index in range(steps):
        temperature = math.exp(low + (high - low) * index / (steps - 1))
        loss = 0.0
        for row, valid in zip(logits, targets, strict=True):
            values = torch.tensor(row, dtype=torch.float64) / temperature
            probabilities = values.softmax(0)
            loss -= math.log(max(float(probabilities[valid].sum().item()), 1e-12))
        best = min(best, (loss / len(logits), temperature))
    return best[1]


def fit_temperatures(
    heads: Any,
    extractor: FeatureExtractor,
    dataset: RlcdDataset,
    config: dict[str, Any],
) -> dict[str, float]:
    heads.eval()
    collected: dict[str, tuple[list[list[float]], list[list[int]]]] = {
        task: ([], []) for task in TASKS
    }
    with torch.no_grad():
        for sample in dataset:
            task = str(sample["task_type"])
            question, candidates, mask = extractor(sample)
            logits, _ = heads(task, question, candidates, mask)
            if task == "noul":
                binary_logit = float(logits.item())
                row = [0.0, binary_logit]
                targets = [int(bool(sample["target"]))]
            else:
                row = logits[0].float().cpu().tolist()
                targets = _target_indices(sample)
            collected[task][0].append(row)
            collected[task][1].append(targets)
    return {
        task: _temperature_grid(rows, targets, config)
        for task, (rows, targets) in collected.items()
    }


def calibrated_probabilities(task: str, logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits.float() / temperature
    if task == "noul":
        return torch.sigmoid(scaled)
    if task in {"choice", "score"}:
        return torch.softmax(scaled, dim=-1)
    raise ValueError(f"unsupported task: {task}")


def select_confidence_thresholds(
    records: list[dict[str, Any]],
    *,
    target_accuracy: float = 0.95,
    minimum_accepted: int = 25,
) -> dict[str, Any]:
    """Choose maximum-coverage per-task thresholds on the frozen threshold role."""
    if not 0 < target_accuracy <= 1:
        raise ValueError("target_accuracy must be in (0, 1]")
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    result: dict[str, Any] = {
        "target_accuracy": target_accuracy,
        "minimum_accepted": minimum_accepted,
        "by_task": {},
    }
    for task in TASKS:
        rows = sorted(
            (row for row in records if row["task_type"] == task),
            key=lambda row: float(row["confidence"]),
            reverse=True,
        )
        correct = 0
        best: tuple[int, float, float] | None = None
        for count, row in enumerate(rows, 1):
            correct += int(bool(row["correct"]))
            accuracy = correct / count
            if count >= minimum_accepted and accuracy >= target_accuracy:
                best = (count, float(row["confidence"]), accuracy)
        if best is None:
            accepted, threshold, accuracy = 0, 1.000001, None
        else:
            accepted, threshold, accuracy = best
        result["by_task"][task] = {
            "threshold": threshold,
            "accepted": accepted,
            "questions": len(rows),
            "coverage": accepted / len(rows) if rows else 0.0,
            "accuracy": accuracy,
        }
    return result


def _threshold_report(
    records: list[dict[str, Any]], thresholds: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in TASKS:
        rows = [row for row in records if row["task_type"] == task]
        threshold = float(thresholds["by_task"][task]["threshold"])
        accepted = [row for row in rows if float(row["confidence"]) >= threshold]
        result[task] = {
            "threshold": threshold,
            "questions": len(rows),
            "accepted": len(accepted),
            "coverage": len(accepted) / len(rows) if rows else 0.0,
            "accuracy": (
                sum(bool(row["correct"]) for row in accepted) / len(accepted)
                if accepted
                else None
            ),
        }
    return result


def _source_threshold_reports(
    records: list[dict[str, Any]], thresholds: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    sources = sorted({str(row["source"]) for row in records})
    return {
        source: _threshold_report(
            [row for row in records if str(row["source"]) == source], thresholds
        )
        for source in sources
    }


def evaluate_rlcd_checkpoint(
    config_path: Path,
    checkpoint: Path,
    role: str,
    output_path: Path,
    *,
    model_root: Path = DEFAULT_MODEL_CACHE,
    role_manifest: Path | None = None,
    sft_checkpoint: Path | None = None,
    maximum: int | None = None,
    progress_every: int = 100,
    thresholds_path: Path | None = None,
    select_thresholds_path: Path | None = None,
    target_accuracy: float = 0.95,
    minimum_accepted: int = 25,
) -> dict[str, Any]:
    if role not in {"threshold", "audit", "test"}:
        raise ValueError("role must be threshold, audit, or test")
    if maximum is not None and maximum < 1:
        raise ValueError("maximum must be positive or None")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    summary_path = output_path.with_suffix(".summary.json")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(f"RLCD evaluation output already exists: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("RLCD evaluation requires a CUDA device")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sft_checkpoint is not None:
        config["sft_checkpoint"] = str(sft_checkpoint)
    device = torch.device("cuda")
    processor = load_processor(Path(config["model_config"]), model_root)
    base = load_backbone(
        Path(config["model_config"]), model_root=model_root, dtype=torch.bfloat16, use_lora=False
    )
    backbone = PeftModel.from_pretrained(base, Path(config["sft_checkpoint"]))
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    heads = DecisionHeads(
        int(base.config.text_config.hidden_size),
        int(config.get("head_width", 256)),
        int(config.get("head_attention_heads", 4)),
    ).to(device)
    heads.load_state_dict(load_file(checkpoint / "decision_heads.safetensors", device="cuda"))
    heads.eval()
    temperatures = json.loads((checkpoint / "temperatures.json").read_text(encoding="utf-8"))
    manifest = role_manifest or Path(config["role_manifest"])
    dataset = RlcdDataset(
        manifest, role, seed=int(config.get("seed", 43)), maximum=maximum
    )
    extractor = FeatureExtractor(backbone, processor, config, device)
    totals = _new_totals(device)
    source_totals: dict[str, Tensor] = {}
    records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with output_path.open("w", encoding="utf-8") as output, torch.no_grad():
        for index, sample in enumerate(dataset, 1):
            task = str(sample["task_type"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                question, candidates, mask = extractor(sample)
                logits, _ = heads(task, question, candidates, mask)
            probabilities = calibrated_probabilities(task, logits, float(temperatures[task]))
            metrics = _prediction(task, logits, probabilities, sample)
            _record(totals, task, metrics)
            source = str(sample["source"])
            if source not in source_totals:
                source_totals[source] = _new_totals(device)
            _record(source_totals[source], task, metrics)
            if task == "noul":
                positive_probability = float(probabilities.item())
                probability_values = [1.0 - positive_probability, positive_probability]
                predicted: bool | str = bool(probabilities.item() >= 0.5)
                target: bool | list[str] = bool(sample["target"])
            else:
                probability_values = probabilities[0].float().cpu().tolist()
                predicted_index = max(
                    range(len(probability_values)), key=probability_values.__getitem__
                )
                predicted = str(sample["options"][predicted_index]["id"])
                raw_target = sample["target"]
                target = (
                    [str(value) for value in raw_target]
                    if isinstance(raw_target, list)
                    else [str(raw_target)]
                )
            record = {
                "sample_id": sample["sample_id"],
                "root_id": sample["root_id"],
                "source": source,
                "task_type": task,
                "target": target,
                "prediction": predicted,
                "probabilities": probability_values,
                "confidence": metrics["confidence"],
                "correct": bool(metrics["correct"]),
            }
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % progress_every == 0 or index == len(dataset):
                output.flush()
                print(
                    json.dumps(
                        {
                            "role": role,
                            "evaluated": index,
                            "total": len(dataset),
                            "accuracy": float(totals[:, 1].sum() / totals[:, 0].sum()),
                        }
                    ),
                    flush=True,
                )
    report = _report(totals.cpu())
    report.update(
        {
            "role": role,
            "temperatures": temperatures,
            "high_confidence_errors_0_9": sum(
                not row["correct"] and float(row["confidence"]) >= 0.9 for row in records
            ),
            "high_confidence_errors_0_9_by_source": {
                source: sum(
                    str(row["source"]) == source
                    and not row["correct"]
                    and float(row["confidence"]) >= 0.9
                    for row in records
                )
                for source in sorted(source_totals)
            },
            "by_source": {
                source: _report(values.cpu())
                for source, values in sorted(source_totals.items())
            },
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    if thresholds_path is not None:
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
        report["threshold_policy"] = _threshold_report(records, thresholds)
        report["threshold_policy_by_source"] = _source_threshold_reports(
            records, thresholds
        )
    if select_thresholds_path is not None:
        if role != "threshold":
            raise ValueError("confidence thresholds may only be selected on the threshold role")
        thresholds = select_confidence_thresholds(
            records,
            target_accuracy=target_accuracy,
            minimum_accepted=minimum_accepted,
        )
        select_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        select_thresholds_path.write_text(
            json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report["selected_thresholds"] = thresholds
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def train_rlcd(
    config_path: Path,
    output_dir: Path,
    *,
    model_root: Path = DEFAULT_MODEL_CACHE,
    sft_checkpoint: Path | None = None,
    train_manifest: Path | None = None,
    role_manifest: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sft_checkpoint is not None:
        config["sft_checkpoint"] = str(sft_checkpoint)
    if train_manifest is not None:
        config["train_manifest"] = str(train_manifest)
    if role_manifest is not None:
        config["role_manifest"] = str(role_manifest)
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="bf16" if config.get("bf16", True) else "no",
        kwargs_handlers=[ddp],
    )
    _set_process_name(os.environ.get("VISION_JEV_PROCESS_NAME", ""))
    validate_rlcd_config(config, accelerator.num_processes)
    if output_dir.exists():
        raise FileExistsError(f"RLCD output already exists: {output_dir}")
    seed = int(config.get("seed", 43))
    random.seed(seed + accelerator.process_index)
    torch.manual_seed(seed + accelerator.process_index)

    processor = load_processor(Path(config["model_config"]), model_root)
    base = load_backbone(
        Path(config["model_config"]), model_root=model_root, dtype=torch.bfloat16, use_lora=False
    )
    backbone = PeftModel.from_pretrained(base, Path(config["sft_checkpoint"]))
    backbone.to(accelerator.device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    hidden_size = int(base.config.text_config.hidden_size)
    heads = DecisionHeads(
        hidden_size,
        int(config.get("head_width", 256)),
        int(config.get("head_attention_heads", 4)),
    )
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    train_data = RlcdDataset(
        Path(config["train_manifest"]),
        "train",
        seed=seed,
        maximum=config.get("train_maximum"),
    )
    dev_preview = RlcdDataset(
        Path(config["role_manifest"]),
        "dev",
        seed=seed,
        maximum=int(config.get("dev_preview_questions", 300)),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        collate_fn=_one_row,
        num_workers=int(config.get("dataloader_workers", 0)),
        generator=torch.Generator().manual_seed(seed),
    )
    dev_loader = DataLoader(dev_preview, batch_size=1, collate_fn=_one_row)
    updates_per_epoch = math.ceil(
        len(train_data) / int(config["global_batch_questions"])
    )
    total_steps = int(config.get("max_steps") or updates_per_epoch * int(config["epochs"]))
    scheduler_steps = total_steps * accelerator.num_processes
    warmup_steps = max(
        1, round(scheduler_steps * float(config.get("warmup_ratio", 0.03)))
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, scheduler_steps)
    heads, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        heads, optimizer, train_loader, dev_loader, scheduler
    )
    extractor = FeatureExtractor(backbone, processor, config, accelerator.device)
    output_dir.mkdir(parents=True, exist_ok=False) if accelerator.is_main_process else None
    accelerator.wait_for_everyone()
    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()
    step = 0
    first_loss: float | None = None
    latest_loss = 0.0
    checkpoint_every = int(config.get("checkpoint_every_steps", 500))
    dev_every = int(config.get("dev_every_steps", 500))
    log_every = int(config.get("log_every", 10))

    def emit(payload: dict[str, Any]) -> None:
        if accelerator.is_main_process:
            line = json.dumps(payload, ensure_ascii=False)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    emit(
        {
            "event": "start",
            "train_questions": len(train_data),
            "dev_preview_questions": len(dev_preview),
            "steps": total_steps,
            "world_size": accelerator.num_processes,
            "global_batch_questions": int(config["global_batch_questions"]),
        }
    )
    heads.train()
    for epoch in range(int(config["epochs"])):
        for sample in train_loader:
            with accelerator.accumulate(heads):
                task = str(sample["task_type"])
                with torch.no_grad(), accelerator.autocast():
                    question, candidates, mask = extractor(sample)
                with accelerator.autocast():
                    logits, probs = heads(task, question, candidates, mask)
                    loss = decision_loss(task, logits, probs, sample)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(heads.parameters(), float(config["gradient_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            step += 1
            latest_loss = float(accelerator.reduce(loss.detach(), reduction="mean").item())
            first_loss = latest_loss if first_loss is None else first_loss
            if step % log_every == 0 or step == 1:
                emit(
                    {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "train_loss": latest_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
            if dev_every and step % dev_every == 0:
                report = evaluate_heads(heads, extractor, dev_loader, accelerator)
                emit({"event": "dev_preview", "step": step, **report})
            if checkpoint_every and step % checkpoint_every == 0:
                _save_heads(
                    output_dir / f"checkpoint-step-{step:06d}",
                    accelerator=accelerator,
                    heads=heads,
                    state={"step": step, "epoch": epoch},
                    config=config,
                )
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    full_dev = RlcdDataset(
        Path(config["role_manifest"]),
        "dev",
        seed=seed,
        maximum=config.get("dev_maximum"),
    )
    full_dev_loader = accelerator.prepare(
        DataLoader(full_dev, batch_size=1, collate_fn=_one_row)
    )
    dev_report = evaluate_heads(heads, extractor, full_dev_loader, accelerator)
    _save_heads(
        output_dir / "checkpoint-last",
        accelerator=accelerator,
        heads=heads,
        state={"step": step, "epoch": int(config["epochs"]) - 1},
        config=config,
    )
    temperatures: dict[str, float] = {}
    if accelerator.is_main_process:
        calibration = RlcdDataset(
            Path(config["role_manifest"]),
            "calibration",
            seed=seed,
            maximum=config.get("calibration_maximum"),
        )
        temperatures = fit_temperatures(
            accelerator.unwrap_model(heads), extractor, calibration, config
        )
        (output_dir / "checkpoint-last" / "temperatures.json").write_text(
            json.dumps(temperatures, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    elapsed = time.monotonic() - started
    summary = {
        "steps": step,
        "train_questions": len(train_data),
        "dev_questions": len(full_dev),
        "calibration_questions": int(config["calibration_questions"]),
        "first_train_loss": first_loss,
        "final_train_loss": latest_loss,
        "dev": dev_report,
        "temperatures": temperatures if accelerator.is_main_process else None,
        "elapsed_seconds": elapsed,
        "world_size": accelerator.num_processes,
    }
    if accelerator.is_main_process:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary if accelerator.is_main_process else {}
