"""Load local API credentials without exposing them in repository configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_API_KEYS_PATH = Path("configs/local/api_keys.toml")


@dataclass(frozen=True)
class APIKeys:
    kimi: str = field(repr=False)
    glm: str = field(repr=False)


def load_api_keys(
    path: Path = DEFAULT_API_KEYS_PATH,
    *,
    required_providers: tuple[str, ...] = ("kimi", "glm"),
) -> APIKeys:
    """Read the two supported provider keys from the ignored local TOML file."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    expected = {"kimi": {"api_key"}, "glm": {"api_key"}}
    actual = {
        provider: set(section) if isinstance(section, dict) else set()
        for provider, section in payload.items()
    }
    if actual != expected:
        raise ValueError("api_keys.toml must contain only kimi.api_key and glm.api_key")

    keys = APIKeys(
        kimi=str(payload["kimi"]["api_key"]).strip(), glm=str(payload["glm"]["api_key"]).strip()
    )
    unknown = sorted(set(required_providers) - set(expected))
    if unknown:
        raise ValueError(f"unknown required API providers: {unknown}")
    missing = [provider for provider in required_providers if not getattr(keys, provider)]
    if missing:
        raise ValueError(f"fill required API keys: {', '.join(missing)}")
    return keys
