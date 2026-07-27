"""
Run persistence.

Regression detection is impossible without history, so every evaluation pass
is written to disk as a versioned run. Storage is plain JSON on the local
filesystem -- deliberately boring, no database to stand up, and the artefacts
are diffable and committable.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .models import EvalRun


def _run_path(run_id: str) -> Path:
    return config.RUNS_DIR / f"{run_id}.json"


def new_run_id(version: str, eval_type: str) -> str:
    """Timestamped, sortable, collision-resistant run id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    safe_version = version.replace("/", "-").replace(" ", "_")
    return f"{stamp}_{eval_type}_{safe_version}_{suffix}"


def save_run(run: EvalRun) -> Path:
    """Write a run to disk and return its path."""
    path = _run_path(run.run_id)
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return path


def load_run(run_id: str) -> EvalRun | None:
    path = _run_path(run_id)
    if not path.exists():
        return None
    return EvalRun.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_runs(eval_type: str | None = None) -> list[EvalRun]:
    """All stored runs, newest first, optionally filtered by eval type."""
    runs: list[EvalRun] = []
    for path in config.RUNS_DIR.glob("*.json"):
        try:
            run = EvalRun.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, KeyError):
            continue  # skip corrupt files rather than failing the listing
        if eval_type and run.eval_type != eval_type:
            continue
        runs.append(run)

    return sorted(runs, key=lambda r: r.created_at, reverse=True)


def latest_run(
    eval_type: str,
    exclude_run_id: str | None = None,
) -> EvalRun | None:
    """
    Most recent run of a given type -- the default regression baseline.

    `exclude_run_id` lets a freshly saved run compare against the one before
    it rather than against itself.
    """
    for run in list_runs(eval_type):
        if exclude_run_id and run.run_id == exclude_run_id:
            continue
        return run
    return None


def find_run_by_version(eval_type: str, version: str) -> EvalRun | None:
    """Most recent run matching a specific version label."""
    for run in list_runs(eval_type):
        if run.version == version:
            return run
    return None


def delete_run(run_id: str) -> bool:
    path = _run_path(run_id)
    if path.exists():
        path.unlink()
        return True
    return False
