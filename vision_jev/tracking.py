"""Immutable, dependency-free experiment run records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(root: Path) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_revision": _git_revision(root),
    }


def create_run(root: Path, config_path: Path, data_path: Path, run_id: str | None = None) -> Path:
    config_path = config_path.resolve()
    data_path = data_path.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(f"{stamp}:{config_path}:{data_path}".encode()).hexdigest()[:8]
    run_id = run_id or f"{stamp}-{suffix}"
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)
    frozen_config = run_dir / "config.json"
    frozen_data = run_dir / f"data{data_path.suffix}"
    shutil.copyfile(config_path, frozen_config)
    shutil.copyfile(data_path, frozen_data)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "created",
        "evidence_status": "planned",
        "created_at": _now(),
        "config": {
            "source": str(config_path),
            "path": frozen_config.name,
            "sha256": sha256_file(frozen_config),
        },
        "data": {
            "source": str(data_path),
            "path": frozen_data.name,
            "sha256": sha256_file(frozen_data),
        },
        "environment": environment(root),
        "result": None,
    }
    (run_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "commands.log").write_text("", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    return run_dir


def finalize_run(run_dir: Path, status: str, summary: str, evidence_status: str) -> None:
    allowed_status = {"completed", "failed", "aborted"}
    allowed_evidence = {"structural_check", "measured", "reproduced", "released"}
    if status not in allowed_status:
        raise ValueError(f"status must be one of {sorted(allowed_status)}")
    if evidence_status not in allowed_evidence:
        raise ValueError(f"evidence_status must be one of {sorted(allowed_evidence)}")
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["status"] != "created" and record["status"] != "running":
        raise ValueError(f"run is already finalized with status {record['status']}")
    record["status"] = status
    record["evidence_status"] = evidence_status
    record["finished_at"] = _now()
    record["result"] = summary
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
