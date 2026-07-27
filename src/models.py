"""
Data models for the evaluation platform.

Two levels of evaluation are supported:

1. Response-level  -- a single question/answer pair (classic LLM eval)
2. Trace-level     -- a multi-step agent run (agentic eval)

Trace-level is the harder problem: an agent can produce a correct final
answer through a wrong process (lucky guess), or fail despite reasoning
correctly (bad tool call). Only inspecting the trace catches both.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------
# Response-level evaluation
# --------------------------------------------------------------------------

class RootCause(str, Enum):
    """Constrained root-cause taxonomy. Keeps judge output analysable."""

    NO_ISSUE = "No Issue"
    KNOWLEDGE_GAP = "Knowledge Gap"
    POTENTIAL_HALLUCINATION = "Potential Hallucination"
    LOW_CONFIDENCE = "Low Confidence"
    FACTUAL_ERROR = "Factual Error"
    EVALUATION_ERROR = "Evaluation Error"


class Recommendation(str, Enum):
    """Constrained recommendation taxonomy -> maps to a concrete action."""

    NO_ACTION = "No Action Required"
    IMPROVE_KNOWLEDGE_BASE = "Improve Knowledge Base"
    ADD_RETRIEVAL_LAYER = "Add Retrieval Layer"
    HUMAN_REVIEW = "Human Review Recommended"
    RETRAIN_MODEL = "Retrain Model"
    MANUAL_REVIEW = "Manual Review Required"


@dataclass
class ResponseScore:
    """Judge output for a single response, plus the composite score."""

    relevance: float
    completeness: float
    truthfulness: float
    hallucination: float          # higher = worse
    safety: float
    root_cause: str
    recommendation: str
    reasoning: str
    quality_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.quality_score:
            self.quality_score = self.compute_quality_score()

    def compute_quality_score(self) -> float:
        """Composite score. Hallucination is inverted since higher is worse."""
        return round(
            (
                self.relevance
                + self.completeness
                + self.truthfulness
                + self.safety
                + (10 - self.hallucination)
            )
            / 5,
            2,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Trace-level (agent) evaluation
# --------------------------------------------------------------------------

class StepType(str, Enum):
    """The kinds of steps an agent run can contain."""

    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ESCALATION = "escalation"      # handed off to a human
    REFUSAL = "refusal"            # declined the task


@dataclass
class TraceStep:
    """One step inside an agent run."""

    step: int
    type: StepType
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TraceStep:
        return cls(
            step=int(raw["step"]),
            type=StepType(raw["type"]),
            content=raw.get("content", "") or "",
            tool_name=raw.get("tool_name"),
            tool_args=raw.get("tool_args") or {},
            tool_result=raw.get("tool_result"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload


@dataclass
class AgentTrace:
    """A complete agent run for one task."""

    task_id: str
    task: str
    steps: list[TraceStep] = field(default_factory=list)
    final_answer: str | None = None
    cost_usd: float = 0.0
    latency_ms: int = 0
    agent_version: str = "unversioned"

    # ---- derived views used by the metrics layer ----

    @property
    def tool_sequence(self) -> list[str]:
        """Tools actually called, in order."""
        return [
            s.tool_name
            for s in self.steps
            if s.type is StepType.TOOL_CALL and s.tool_name
        ]

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def escalated(self) -> bool:
        return any(
            s.type in (StepType.ESCALATION, StepType.REFUSAL)
            for s in self.steps
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentTrace:
        return cls(
            task_id=raw["task_id"],
            task=raw["task"],
            steps=[TraceStep.from_dict(s) for s in raw.get("steps", [])],
            final_answer=raw.get("final_answer"),
            cost_usd=float(raw.get("cost_usd", 0.0)),
            latency_ms=int(raw.get("latency_ms", 0)),
            agent_version=raw.get("agent_version", "unversioned"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "agent_version": self.agent_version,
        }


@dataclass
class GoldenTask:
    """
    A task with a known-correct outcome.

    `should_refuse` marks guardrail tests -- tasks the agent is expected to
    decline or escalate rather than execute. An agent that cheerfully
    completes a destructive request has failed, even if it executed well.
    """

    task_id: str
    task: str
    expected_outcome: str
    expected_tools: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    should_refuse: bool = False
    min_steps: int = 1
    category: str = "general"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GoldenTask:
        return cls(
            task_id=raw["task_id"],
            task=raw["task"],
            expected_outcome=raw.get("expected_outcome", ""),
            expected_tools=raw.get("expected_tools", []),
            available_tools=raw.get("available_tools", []),
            should_refuse=bool(raw.get("should_refuse", False)),
            min_steps=int(raw.get("min_steps", 1)),
            category=raw.get("category", "general"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceScore:
    """
    Evaluation result for one agent trace.

    Deterministic metrics are computed from the trace directly -- no LLM
    involved, so they are cheap, fast, and reproducible. Judge metrics are
    optional and only assess things code cannot: reasoning quality and
    groundedness.
    """

    task_id: str

    # deterministic
    task_success: bool = False
    tool_correctness: float = 0.0        # 0..1 sequence match
    tool_hallucination: bool = False     # called a non-existent tool
    step_efficiency: float = 0.0         # 0..1, min_steps / actual
    escalation_correct: bool = True
    cost_usd: float = 0.0
    latency_ms: int = 0

    # judge (optional)
    reasoning_quality: float | None = None
    groundedness: float | None = None
    judge_notes: str = ""

    failure_modes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """A trace passes only if it succeeded and broke no hard rules."""
        return (
            self.task_success
            and not self.tool_hallucination
            and self.escalation_correct
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


# --------------------------------------------------------------------------
# Runs -- the unit of regression tracking
# --------------------------------------------------------------------------

@dataclass
class EvalRun:
    """
    One complete evaluation pass, persisted so later runs can be compared
    against it. This is what makes regression detection possible.
    """

    run_id: str
    version: str
    eval_type: str                       # "response" | "trace"
    created_at: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalRun:
        return cls(
            run_id=raw["run_id"],
            version=raw.get("version", "unversioned"),
            eval_type=raw.get("eval_type", "response"),
            created_at=raw.get("created_at", ""),
            summary=raw.get("summary", {}),
            records=raw.get("records", []),
            notes=raw.get("notes", ""),
        )
