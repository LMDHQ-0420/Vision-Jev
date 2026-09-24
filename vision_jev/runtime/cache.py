"""Content-addressed identity for reusable model state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CacheIdentity:
    image_sha256: str
    text_token_sha256: str
    processor_revision: str
    processor_config_sha256: str
    model_revision: str
    adapter_revision: str
    mode: str
    crop_or_augmentation_sha256: str

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
