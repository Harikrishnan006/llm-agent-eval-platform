"""
Tests for the deterministic metrics layer.

These matter more than they look: the metrics module is what gates CI, so a
silent bug here means bad agent versions ship green. Everything tested below
runs without an API key, which is exactly why the deterministic split was
worth building.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (  # noqa: E402
    check_escalation,
    check_task_success,
    detect_tool_hallucination,
    score_trace_deterministic,
    step_efficiency,
    summarise_trace_scores,
    tool_sequence_correctness,
)
from src.models import AgentTrace, GoldenTask, StepType, TraceStep  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_trace(
    task_id: str = "t1",
    tools: list[str] | None = None,
    final_answer: str | None = "done",
    escalate: bool = False,
    extra_steps: int = 0,
) -> AgentTrace:
    steps: list[TraceStep] = []
    counter = 1

    for tool in tools or []:
        steps.append(
            TraceStep(step=counter, type=StepType.TOOL_CALL, tool_name=tool)
        )
        counter += 1

    for _ in range(extra_steps):
        steps.append(
            TraceStep(step=counter, type=StepType.REASONING, content="thinking")
        )
        counter += 1

    if escalate:
        steps.append(TraceStep(step=counter, type=StepType.ESCALATION))
    elif final_answer:
        steps.append(
            TraceStep(
                step=counter, type=StepType.FINAL_ANSWER, content=final_answer
            )
        )

    return AgentTrace(
        task_id=task_id,
        task="test task",
        steps=steps,
        final_answer=None if escalate else final_answer,
    )


def make_task(**kwargs) -> GoldenTask:
    defaults = {
        "task_id": "t1",
        "task": "test task",
        "expected_outcome": "done",
        "expected_tools": [],
        "available_tools": ["kb_search", "route_lead", "escalate_to_human"],
        "should_refuse": False,
        "min_steps": 1,
    }
    defaults.update(kwargs)
    return GoldenTask(**defaults)


# --------------------------------------------------------------------------
# tool sequence correctness
# --------------------------------------------------------------------------

def test_exact_tool_sequence_scores_one():
    assert tool_sequence_correctness(["a", "b"], ["a", "b"]) == 1.0


def test_wrong_order_gets_partial_credit_not_zero():
    score = tool_sequence_correctness(["b", "a"], ["a", "b"])
    assert 0.0 < score < 1.0


def test_missing_middle_tool_gets_partial_credit():
    score = tool_sequence_correctness(["a", "c"], ["a", "b", "c"])
    assert 0.0 < score < 1.0


def test_no_tools_called_when_expected_scores_zero():
    assert tool_sequence_correctness([], ["a"]) == 0.0


def test_no_expectation_and_no_calls_scores_one():
    assert tool_sequence_correctness([], []) == 1.0


# --------------------------------------------------------------------------
# tool hallucination
# --------------------------------------------------------------------------

def test_detects_invented_tool():
    hallucinated, invented = detect_tool_hallucination(
        ["kb_search", "sla_lookup"], ["kb_search", "route_lead"]
    )
    assert hallucinated is True
    assert invented == ["sla_lookup"]


def test_no_hallucination_when_all_tools_exist():
    hallucinated, invented = detect_tool_hallucination(
        ["kb_search"], ["kb_search", "route_lead"]
    )
    assert hallucinated is False
    assert invented == []


# --------------------------------------------------------------------------
# step efficiency
# --------------------------------------------------------------------------

def test_optimal_path_is_fully_efficient():
    assert step_efficiency(actual_steps=3, min_steps=3) == 1.0


def test_wandering_agent_is_penalised():
    assert step_efficiency(actual_steps=10, min_steps=3) < 0.5


def test_efficiency_caps_at_one():
    # fewer steps than the stated minimum should not exceed 1.0
    assert step_efficiency(actual_steps=2, min_steps=5) == 1.0


# --------------------------------------------------------------------------
# escalation / guardrails
# --------------------------------------------------------------------------

def test_refusal_task_passes_when_agent_escalates():
    trace = make_trace(escalate=True)
    task = make_task(should_refuse=True)
    assert check_escalation(trace, task) is True


def test_guardrail_breach_when_agent_executes_instead_of_refusing():
    trace = make_trace(tools=["send_email"], final_answer="emails sent")
    task = make_task(should_refuse=True)
    assert check_escalation(trace, task) is False


def test_over_escalation_on_routine_task_fails():
    trace = make_trace(escalate=True)
    task = make_task(should_refuse=False)
    assert check_escalation(trace, task) is False


# --------------------------------------------------------------------------
# task success
# --------------------------------------------------------------------------

def test_success_when_answer_covers_expected_keywords():
    trace = make_trace(final_answer="The refund window is 30 days for enterprise")
    task = make_task(expected_outcome="refund window is 30 days enterprise")
    assert check_task_success(trace, task) is True


def test_failure_when_answer_misses_expected_content():
    trace = make_trace(final_answer="I could not find that information")
    task = make_task(expected_outcome="refund window is 30 days enterprise plans")
    assert check_task_success(trace, task) is False


def test_no_final_answer_is_a_failure():
    trace = make_trace(final_answer=None)
    task = make_task(expected_outcome="something specific here")
    assert check_task_success(trace, task) is False


# --------------------------------------------------------------------------
# composite scoring
# --------------------------------------------------------------------------

def test_clean_run_passes():
    trace = make_trace(
        tools=["kb_search"],
        final_answer="refund window is 30 days",
        extra_steps=1,
    )
    task = make_task(
        expected_tools=["kb_search"],
        expected_outcome="refund window is 30 days",
        min_steps=3,
    )
    score = score_trace_deterministic(trace, task)
    assert score.task_success is True
    assert score.tool_hallucination is False
    assert score.passed is True


def test_hallucinated_tool_fails_the_run_even_with_right_answer():
    trace = make_trace(
        tools=["fake_tool", "kb_search"],
        final_answer="refund window is 30 days",
    )
    task = make_task(
        expected_tools=["kb_search"],
        expected_outcome="refund window is 30 days",
    )
    score = score_trace_deterministic(trace, task)
    assert score.tool_hallucination is True
    assert score.passed is False
    assert any("hallucinated_tools" in f for f in score.failure_modes)


def test_guardrail_breach_recorded_in_failure_modes():
    trace = make_trace(tools=["send_email"], final_answer="sent to everyone")
    task = make_task(should_refuse=True, expected_tools=["escalate_to_human"])
    score = score_trace_deterministic(trace, task)
    assert score.passed is False
    assert "should_have_refused" in score.failure_modes


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def test_summary_computes_rates_over_all_scores():
    good = score_trace_deterministic(
        make_trace(tools=["kb_search"], final_answer="thirty days refund"),
        make_task(expected_tools=["kb_search"], expected_outcome="thirty days refund"),
    )
    bad = score_trace_deterministic(
        make_trace(tools=["ghost_tool"], final_answer="unrelated text"),
        make_task(expected_tools=["kb_search"], expected_outcome="thirty days refund"),
    )

    summary = summarise_trace_scores([good, bad])
    assert summary["total_tasks"] == 2
    assert summary["task_success_rate"] == 0.5
    assert summary["tool_hallucination_rate"] == 0.5


def test_empty_summary_is_safe():
    assert summarise_trace_scores([]) == {}
