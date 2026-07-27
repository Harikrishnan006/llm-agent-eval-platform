#!/usr/bin/env python3
"""
CLI evaluation runner.

Designed for CI: exits non-zero when a regression is detected, so a quality
drop fails the build before it reaches users. This is the difference between
"we have evaluation" and "evaluation actually gates our releases".

Examples
--------
    # trace eval, deterministic only, compare to previous run
    python scripts/run_eval.py --type trace --version v2

    # response eval with the LLM judge on 20 records
    python scripts/run_eval.py --type response --version v2 --sample-size 20

    # compare against a specific earlier version
    python scripts/run_eval.py --type trace --version v3 --baseline v1

    # first run, nothing to compare against yet
    python scripts/run_eval.py --type trace --version v1 --no-compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agent_eval, config, llm_eval, regression, storage  # noqa: E402
from src.metrics import summarise_trace_scores  # noqa: E402


def _progress(current: int, total: int) -> None:
    if total and (current == total or current % 5 == 0):
        print(f"  evaluated {current}/{total}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an evaluation pass and check for regressions."
    )
    parser.add_argument(
        "--type",
        choices=["response", "trace"],
        default="trace",
        help="response = single Q/A pairs, trace = multi-step agent runs",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="version label for this run, e.g. v2 or a git sha",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=config.DEFAULT_SAMPLE_SIZE,
        help="records to evaluate (response type only)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="enable the LLM judge (requires GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="version to compare against (default: most recent run)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="skip regression check, e.g. when establishing a first baseline",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="write the regression report to this path as JSON",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="free-text note stored with the run",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    use_judge = args.judge
    if use_judge and not config.judge_available():
        print(
            "WARNING: --judge requested but GEMINI_API_KEY is not set. "
            "Continuing with deterministic metrics only.",
            file=sys.stderr,
        )
        use_judge = False

    print(f"Running {args.type} evaluation  version={args.version}")

    # ---- run the evaluation ------------------------------------------------
    if args.type == "trace":
        tasks = agent_eval.load_golden_tasks()
        traces = agent_eval.load_traces()
        if not tasks or not traces:
            print(
                "ERROR: golden tasks or traces missing from data/.",
                file=sys.stderr,
            )
            return 2

        print(f"  {len(traces)} traces against {len(tasks)} golden tasks")
        run, scores = agent_eval.run_agent_eval(
            version=args.version,
            use_judge=use_judge,
            progress=_progress,
            notes=args.notes,
        )

        guardrails = agent_eval.guardrail_results(scores, tasks)
        failures = agent_eval.failure_mode_breakdown(scores)
    else:
        run, results = llm_eval.run_response_eval(
            version=args.version,
            sample_size=args.sample_size,
            progress=_progress,
            use_judge=use_judge,
            notes=args.notes,
        )
        guardrails = None
        failures = None
        scores = None

    # ---- print the summary -------------------------------------------------
    print("\nSummary")
    print("-------")
    for key, value in run.summary.items():
        print(f"  {key:<26} {value}")

    if guardrails and guardrails["total"]:
        print(
            f"\nGuardrail tasks: {guardrails['correct']}/{guardrails['total']} "
            "handled correctly"
        )
        if guardrails["breaches"]:
            print(f"  BREACHES: {', '.join(guardrails['breaches'])}")

    if failures:
        print("\nFailure modes")
        print("-------------")
        for mode, count in failures.items():
            print(f"  {mode:<26} {count}")

    print(f"\nRun saved: {run.run_id}")

    # ---- regression check --------------------------------------------------
    if args.no_compare:
        print("\nRegression check skipped (--no-compare).")
        return 0

    if args.baseline:
        baseline = storage.find_run_by_version(args.type, args.baseline)
        if baseline is None:
            print(
                f"\nNo run found for baseline version '{args.baseline}'. "
                "Nothing to compare.",
                file=sys.stderr,
            )
            return 0
    else:
        baseline = storage.latest_run(args.type, exclude_run_id=run.run_id)
        if baseline is None:
            print(
                "\nNo earlier run found -- this run becomes the baseline."
            )
            return 0

    report = regression.compare_runs(baseline, run)
    print("\n" + report.format_report())

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"\nReport written to {args.json_out}")

    # non-zero exit fails the CI job
    return 1 if report.has_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
