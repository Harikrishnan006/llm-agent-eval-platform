"""Central configuration. Everything tunable lives here."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = DATA_DIR / "runs"

GOLDEN_TASKS_PATH = DATA_DIR / "golden_tasks.json"
SAMPLE_TRACES_PATH = DATA_DIR / "sample_traces.json"
TRUTHFULQA_PATH = DATA_DIR / "truthfulqa.parquet"

RUNS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Model / API
# --------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")

# Low temperature: an evaluator must be reproducible, not creative.
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.1"))

# Rough per-call cost estimate, only used for cost reporting in demos.
COST_PER_JUDGE_CALL_USD = float(os.getenv("COST_PER_JUDGE_CALL_USD", "0.0002"))


# --------------------------------------------------------------------------
# Evaluation defaults
# --------------------------------------------------------------------------

DEFAULT_SAMPLE_SIZE = 10
MAX_SAMPLE_SIZE = 100

# Flag a response as high-hallucination at or above this score.
HALLUCINATION_ALERT_THRESHOLD = 7.0

# Route to human review below this composite quality score.
HUMAN_REVIEW_QUALITY_THRESHOLD = 6.0


# --------------------------------------------------------------------------
# Regression gates
# --------------------------------------------------------------------------
# A metric regresses when it moves in the wrong direction by more than the
# tolerance below. "higher_is_better" controls the direction.

REGRESSION_RULES: dict[str, dict[str, float | bool]] = {
    # response-level
    "avg_quality_score": {"tolerance": 0.3, "higher_is_better": True},
    "avg_truthfulness": {"tolerance": 0.3, "higher_is_better": True},
    "avg_hallucination": {"tolerance": 0.3, "higher_is_better": False},
    # trace-level
    "task_success_rate": {"tolerance": 0.05, "higher_is_better": True},
    "tool_correctness": {"tolerance": 0.05, "higher_is_better": True},
    "tool_hallucination_rate": {"tolerance": 0.01, "higher_is_better": False},
    "escalation_accuracy": {"tolerance": 0.01, "higher_is_better": True},
    "avg_steps": {"tolerance": 1.0, "higher_is_better": False},
    "avg_cost_usd": {"tolerance": 0.002, "higher_is_better": False},
}


def judge_available() -> bool:
    """True when an API key is configured, so callers can degrade gracefully."""
    return bool(GEMINI_API_KEY)
