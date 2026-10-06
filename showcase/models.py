"""Lazy GPU adapters for the baseline and Vision-Jev decision model."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from showcase.schema import ModelSpec


@dataclass(frozen=True)
class Decision:
    action: str | None
    probabilities: dict[str, float]
    latency_ms: float
    raw_output: str | None = None


class ModelAdapter(Protocol):
    def decide(self, sample: dict[str, Any]) -> Decision: ...

    def close(self) -> None: ...


class QwenBaseAdapter:
    def __init__(self, spec: ModelSpec, model_root: Path) -> None:
        import torch

        from vision_jev.model.qwen35 import load_backbone, load_processor

        if not torch.cuda.is_available():
            raise RuntimeError("showcase inference requires CUDA")
        self.torch = torch
        self.processor = load_processor(spec.model_config, model_root)
        self.model = load_backbone(
            spec.model_config, model_root=model_root, dtype=torch.bfloat16, use_lora=False
        ).to("cuda").eval()

    def decide(self, sample: dict[str, Any]) -> Decision:
        from vision_jev.train.sft import NativeQwenCollator, conversation

        budget = NativeQwenCollator(self.processor).visual_budget([sample])
        batch = self.processor.apply_chat_template(
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
                        "longest_edge": budget * 16 * 16 * 4,
                    }
                }
            },
        ).to("cuda")
        input_length = int(batch["input_ids"].shape[1])
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.no_grad():
            generated = self.model.generate(
                **batch, max_new_tokens=24, do_sample=False, use_cache=True
            )
        self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        raw = self.processor.decode(
            generated[0, input_length:], skip_special_tokens=True
        ).strip()
        options = {str(option["id"]) for option in sample["options"]}
        action: str | None = None
        try:
            parsed = json.loads(raw)
            candidate = parsed.get("choice") if isinstance(parsed, dict) else None
            if isinstance(candidate, str) and candidate in options:
                action = candidate
        except json.JSONDecodeError:
            pass
        return Decision(action, {}, latency_ms, raw)

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


class VisionJevAdapter:
    def __init__(self, spec: ModelSpec, model_root: Path) -> None:
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file

        from vision_jev.model.qwen35 import load_backbone, load_processor
        from vision_jev.train.rlcd import DecisionHeads, FeatureExtractor

        if not torch.cuda.is_available():
            raise RuntimeError("showcase inference requires CUDA")
        assert spec.sft_checkpoint is not None
        assert spec.rlcd_config is not None
        assert spec.rlcd_checkpoint is not None
        config = json.loads(spec.rlcd_config.read_text(encoding="utf-8"))
        processor = load_processor(spec.model_config, model_root)
        base = load_backbone(
            spec.model_config, model_root=model_root, dtype=torch.bfloat16, use_lora=False
        )
        self.model = PeftModel.from_pretrained(base, spec.sft_checkpoint).to("cuda").eval()
        self.heads = DecisionHeads(
            int(base.config.text_config.hidden_size),
            int(config.get("head_width", 256)),
            int(config.get("head_attention_heads", 4)),
        ).to("cuda")
        self.heads.load_state_dict(
            load_file(spec.rlcd_checkpoint / "decision_heads.safetensors", device="cuda")
        )
        self.heads.eval()
        self.temperatures = json.loads(
            (spec.rlcd_checkpoint / "temperatures.json").read_text(encoding="utf-8")
        )
        self.extractor = FeatureExtractor(self.model, processor, config, torch.device("cuda"))
        self.torch = torch

    def decide(self, sample: dict[str, Any]) -> Decision:
        from vision_jev.train.rlcd import calibrated_probabilities

        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.no_grad(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16
        ):
            question, candidates, mask = self.extractor(sample)
            logits, _ = self.heads("choice", question, candidates, mask)
        values = calibrated_probabilities(
            "choice", logits, float(self.temperatures["choice"])
        )[0].float().cpu().tolist()
        self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        options = [str(option["id"]) for option in sample["options"]]
        probabilities = dict(zip(options, values, strict=True))
        action = max(probabilities, key=probabilities.__getitem__)
        return Decision(action, probabilities, latency_ms)

    def close(self) -> None:
        del self.extractor, self.heads, self.model
        self.torch.cuda.empty_cache()


def load_adapter(spec: ModelSpec, model_root: Path) -> ModelAdapter:
    if spec.kind == "qwen_base":
        return QwenBaseAdapter(spec, model_root)
    if spec.kind == "vision_jev_rlcd":
        return VisionJevAdapter(spec, model_root)
    raise ValueError(f"unsupported model kind: {spec.kind}")
