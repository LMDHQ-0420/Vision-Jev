"""Pinned Qwen3.5 native processor and multimodal backbone integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

DEFAULT_MODEL_CACHE = Path("/mnt/sda1/sol_data/vision-jev/models")


def load_model_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_id", "model_revision", "processor_revision"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"model config is missing: {', '.join(missing)}")
    return cast(dict[str, Any], config)


def local_snapshot_path(config: dict[str, Any], model_root: Path = DEFAULT_MODEL_CACHE) -> Path:
    name = str(config["model_id"]).replace("/", "--")
    return model_root / name / str(config["model_revision"])


def prepare_snapshot(config_path: Path, model_root: Path = DEFAULT_MODEL_CACHE) -> Path:
    """Explicitly download the pinned model snapshot outside model construction."""
    from huggingface_hub import snapshot_download

    config = load_model_config(config_path)
    destination = local_snapshot_path(config, model_root)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(config["model_id"]),
        revision=str(config["model_revision"]),
        local_dir=destination,
    )
    return destination


def load_processor(config_path: Path, model_root: Path = DEFAULT_MODEL_CACHE) -> Any:
    from transformers import AutoProcessor

    config = load_model_config(config_path)
    snapshot = local_snapshot_path(config, model_root)
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"model snapshot is not prepared: {snapshot}")
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)  # type: ignore[no-untyped-call]
    processor.tokenizer.padding_side = "right"
    return processor


def load_backbone(
    config_path: Path,
    *,
    model_root: Path = DEFAULT_MODEL_CACHE,
    dtype: Any = None,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Any:
    from transformers import AutoModelForMultimodalLM

    config = load_model_config(config_path)
    snapshot = local_snapshot_path(config, model_root)
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"model snapshot is not prepared: {snapshot}")
    model = AutoModelForMultimodalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=dtype,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj")
        found = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, module in model.named_modules()
                if name.startswith("model.language_model")
                and name.rsplit(".", 1)[-1] in suffixes
                and module.__class__.__name__ == "Linear"
            }
        )
        if not found:
            raise RuntimeError("no audited Qwen3.5 language projection modules were found")
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=found,
            exclude_modules=r".*visual.*",
            bias="none",
        )
        model = get_peft_model(model, lora)
    return model
