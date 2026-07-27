"""
Regression detection.

Compares a run against a baseline and decides whether quality moved backwards
by more than the configured tolerance. Direction awareness matters: a rise in
`avg_hallucination` is a regression while a rise in `task_success_rate` is an
improvement, so each metric declares which way is good in config.

The output is designed to gate a CI pipeline -- `has_regression` maps directly
to a non-zero exit code, so a quality drop can fail a build before it ships.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from . import config
from .models import EvalRun


@dataclass
class MetricDelta:
    """One metric compared across two runs."""

    metric: str
    baseline: float
    current: float
    delta: float
    higher_is_better: bool
    tolerance: float
    status: str          # "improved" | "stable" | "regressed"

    @property
    def regressed(self) -> bool:
        return self.status == "regressed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_line(self) -> str:
        """Human-readable single line, suitable for CI logs."""
        icon = {
            "improved": "[+]",
            "stable": "[=]",
            "regressed": "[!]",
        }[self.status]
        arrow = "->"
        detail = f"{self.baseline:g} {arrow} {self.current:g}"
        sign = "+" if self.delta > 0 else ""
        return (
            f"{icon} {self.metric:<26} {detail:<22} "
            f"({sign}{self.delta:g})"
            + ("  REGRESSION" if self.regressed else "")
        )


@dataclass
class RegressionReport:
    """Full comparison between a baseline run and a current run."""

    baseline_run_id: str
    baseline_version: str
    current_run_id: str
    current_version: str
    deltas: list[MetricDelta]

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.regressed]

    @property
    def improvements(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.status == "improved"]

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "baseline_version": self.baseline_version,
            "current_run_id": self.current_run_id,
            "current_version": self.current_version,
            "has_regression": self.has_regression,
            "regression_count": len(self.regressions),
            "deltas": [d.to_dict() for d in self.deltas],
        }

    def format_report(self) -> str:
        """Block of text for terminal or CI output."""
        header = (
            f"Regression check: {self.baseline_version} "
            f"-> {self.current_version}"
        )
        lines = [header, "=" * len(header)]
        lines.extend(d.format_line() for d in self.deltas)
        lines.append("")

        if self.has_regression:
            names = ", ".join(d.metric for d in self.regressions)
            lines.append(f"FAILED: {len(self.regressions)} regression(s) -- {names}")
        else:
            lines.append("PASSED: no regressions detected")

        return "\n".join(lines)


def _classify(
    baseline: float,
    current: float,
    higher_is_better: bool,
    tolerance: float,
) -> str:
    """Decide whether a metric improved, stayed stable, or regressed."""
    delta = current - baseline

    if abs(delta) <= tolerance:
        return "stable"

    moved_up = delta > 0
    good_direction = moved_up if higher_is_better else not moved_up
    return "improved" if good_direction else "regressed"


def compare_runs(baseline: EvalRun, current: EvalRun) -> RegressionReport:
    """
    Build a regression report between two runs.

    Only metrics present in both runs and declared in REGRESSION_RULES are
    compared -- unknown metrics are reported by the summary view but never
    gate a build, since no tolerance has been agreed for them.
    """
    deltas: list[MetricDelta] = []

    for metric, rule in config.REGRESSION_RULES.items():
        if metric not in baseline.summary or metric not in current.summary:
            continue

        base_val = float(baseline.summary[metric])
        curr_val = float(current.summary[metric])
        higher_is_better = bool(rule["higher_is_better"])
        tolerance = float(rule["tolerance"])

        deltas.append(
            MetricDelta(
                metric=metric,
                baseline=base_val,
                current=curr_val,
                delta=round(curr_val - base_val, 5),
                higher_is_better=higher_is_better,
                tolerance=tolerance,
                status=_classify(
                    base_val, curr_val, higher_is_better, tolerance
                ),
            )
        )

    return RegressionReport(
        baseline_run_id=baseline.run_id,
        baseline_version=baseline.version,
        current_run_id=current.run_id,
        current_version=current.version,
        deltas=deltas,
    )
