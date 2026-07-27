"""
Agent (trace-level) evaluation.

Runs the golden task set against a set of agent traces and scores each one.
Deterministic metrics always run; the LLM judge is optional and only adds
reasoning-quality and groundedness on top.

Keeping the pass/fail gate deterministic is a deliberate choice: a CI pipeline
that fails a build should not depend on a model call that might time out,
return malformed JSON, or score differently on a rerun.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import config, judge, storage
from .metrics import (
    format_trace_for_judge,
    score_trace_deterministic,
    summarise_trace_scores,
)
from .models import AgentTrace, EvalRun, GoldenTask, TraceScore

ProgressFn = Callable[[int, int], None]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_golden_tasks(path: Path | None = None) -> list[GoldenTask]:
    """Load the golden task set -- tasks with known-correct outcomes."""
    path = path or config.GOLDEN_TASKS_PATH
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenTask.from_dict(item) for item in raw]


def load_traces(path: Path | None = None) -> list[AgentTrace]:
    """Load agent traces to be evaluated."""
    path = path or config.SAMPLE_TRACES_PATH
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [AgentTrace.from_dict(item) for item in raw]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_single_trace(
    trace: AgentTrace,
    task: GoldenTask,
    use_judge: bool = False,
) -> TraceScore:
    """Score one trace: deterministic always, judge optionally."""
    score = score_trace_deterministic(trace, task)

    if use_judge:
        verdict = judge.score_trace(
            task=task.task,
            expected_outcome=task.expected_outcome,
            trace_text=format_trace_for_judge(trace),
            final_answer=trace.final_answer,
        )
        score.reasoning_quality = verdict.get("reasoning_quality")
        score.groundedness = verdict.get("groundedness")
        score.judge_notes = verdict.get("notes", "")

    return score


def evaluate_traces(
    traces: list[AgentTrace],
    tasks: list[GoldenTask],
    use_judge: bool = False,
    progress: ProgressFn | None = None,
) -> list[TraceScore]:
    """
    Score every trace that has a matching golden task.

    Traces without a corresponding task are skipped rather than guessed at --
    scoring against an unknown expected outcome would be meaningless.
    """
    task_index = {t.task_id: t for t in tasks}
    scores: list[TraceScore] = []

    matched = [t for t in traces if t.task_id in task_index]
    total = len(matched)

    for index, trace in enumerate(matched, start=1):
        scores.append(
            score_single_trace(
                trace,
                task_index[trace.task_id],
                use_judge=use_judge,
            )
        )
        if progress:
            progress(index, total)

    return scores


def run_agent_eval(
    version: str,
    traces_path: Path | None = None,
    tasks_path: Path | None = None,
    use_judge: bool = False,
    progress: ProgressFn | None = None,
    notes: str = "",
    persist: bool = True,
) -> tuple[EvalRun, list[TraceScore]]:
    """Full trace-level pass: load, score, summarise, persist."""
    tasks = load_golden_tasks(tasks_path)
    traces = load_traces(traces_path)

    scores = evaluate_traces(
        traces, tasks, use_judge=use_judge, progress=progress
    )
    summary = summarise_trace_scores(scores)
    summary["avg_steps"] = round(
        sum(t.step_count for t in traces) / len(traces), 2
    ) if traces else 0.0

    run = EvalRun(
        run_id=storage.new_run_id(version, "trace"),
        version=version,
        eval_type="trace",
        summary=summary,
        records=[s.to_dict() for s in scores],
        notes=notes,
    )

    if persist:
        storage.save_run(run)

    return run, scores


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------

def failure_mode_breakdown(scores: list[TraceScore]) -> dict[str, int]:
    """
    Count occurrences of each failure mode.

    Useful for spotting systemic problems: ten traces all failing with
    `wrong_tool_sequence` points at the tool descriptions or prompt, not at
    ten unrelated bugs.
    """
    counts: dict[str, int] = {}
    for score in scores:
        for mode in score.failure_modes:
            key = mode.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def guardrail_results(
    scores: list[TraceScore],
    tasks: list[GoldenTask],
) -> dict[str, Any]:
    """
    Isolate guardrail (refusal) task performance.

    Reported separately because a single guardrail breach matters more than a
    small dip in average success rate -- an agent that executes a destructive
    request has failed in a way that averages hide.
    """
    refusal_ids = {t.task_id for t in tasks if t.should_refuse}
    relevant = [s for s in scores if s.task_id in refusal_ids]

    if not relevant:
        return {"total": 0, "correct": 0, "breaches": []}

    breaches = [s.task_id for s in relevant if not s.escalation_correct]
    return {
        "total": len(relevant),
        "correct": len(relevant) - len(breaches),
        "breaches": breaches,
    }
