"""
Streamlit UI for the evaluation platform.

Three tabs:
  * Agent Evaluation    -- trace-level scoring against the golden task set
  * Response Evaluation -- classic LLM-as-a-judge scoring on Q/A pairs
  * Regression          -- compare any two stored runs

Nothing runs on page load. Evaluations are triggered by an explicit button and
results are held in session state, so a refresh never silently burns API quota.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import agent_eval, config, llm_eval, regression, storage

st.set_page_config(
    page_title="LLM & Agent Evaluation Platform",
    layout="wide",
)

st.title("LLM & Agent Evaluation Platform")
st.caption(
    "Deterministic agent metrics, LLM-as-a-judge scoring, and "
    "version-to-version regression detection."
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    judge_ready = config.judge_available()
    if judge_ready:
        st.success("Judge model configured")
    else:
        st.warning(
            "No GEMINI_API_KEY found. Deterministic agent metrics still work; "
            "judge-based scoring is disabled."
        )

    version = st.text_input(
        "Version label",
        value="v1",
        help="Stored with the run so later runs can compare against it.",
    )

    st.divider()
    stored = storage.list_runs()
    st.metric("Stored runs", len(stored))
    if stored:
        st.caption(f"Latest: {stored[0].version} ({stored[0].created_at})")


tab_agent, tab_response, tab_regression = st.tabs(
    ["Agent Evaluation", "Response Evaluation", "Regression"]
)


# --------------------------------------------------------------------------
# Tab 1 -- Agent (trace) evaluation
# --------------------------------------------------------------------------

with tab_agent:
    st.subheader("Agent trace evaluation")
    st.markdown(
        "Scores multi-step agent runs against golden tasks. Deterministic "
        "metrics need no API key -- they are computed from the trace itself, "
        "which makes them cheap, fast and reproducible."
    )

    tasks = agent_eval.load_golden_tasks()
    traces = agent_eval.load_traces()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Golden tasks", len(tasks))
    col_b.metric("Traces loaded", len(traces))
    col_c.metric(
        "Guardrail tasks", sum(1 for t in tasks if t.should_refuse)
    )

    use_judge_agent = st.checkbox(
        "Also score reasoning quality with the LLM judge",
        value=False,
        disabled=not judge_ready,
        help="Deterministic metrics run regardless. This adds judgement-quality"
        " scoring, which costs API calls.",
    )

    if st.button("Run agent evaluation", type="primary", key="run_agent"):
        if not tasks or not traces:
            st.error("Golden tasks or traces are missing from data/.")
        else:
            bar = st.progress(0.0, text="Scoring traces...")

            def _progress(done: int, total: int) -> None:
                bar.progress(done / total, text=f"Scored {done}/{total}")

            run, scores = agent_eval.run_agent_eval(
                version=version,
                use_judge=use_judge_agent,
                progress=_progress,
            )
            bar.empty()

            st.session_state["agent_run"] = run
            st.session_state["agent_scores"] = scores
            st.success(f"Run saved: {run.run_id}")

    run = st.session_state.get("agent_run")
    scores = st.session_state.get("agent_scores")

    if run and scores:
        summary = run.summary

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Task success", f"{summary.get('task_success_rate', 0):.0%}")
        m2.metric("Tool correctness", f"{summary.get('tool_correctness', 0):.0%}")
        m3.metric(
            "Tool hallucination",
            f"{summary.get('tool_hallucination_rate', 0):.0%}",
            delta_color="inverse",
        )
        m4.metric(
            "Escalation accuracy",
            f"{summary.get('escalation_accuracy', 0):.0%}",
        )

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Passed", f"{summary.get('passed', 0)}/{summary.get('total_tasks', 0)}")
        m6.metric("Step efficiency", f"{summary.get('avg_step_efficiency', 0):.0%}")
        m7.metric("Avg cost", f"${summary.get('avg_cost_usd', 0):.5f}")
        m8.metric("Avg latency", f"{summary.get('avg_latency_ms', 0):.0f} ms")

        # Guardrails get their own section: one breach matters more than a
        # small dip in an average.
        guardrails = agent_eval.guardrail_results(scores, tasks)
        if guardrails["total"]:
            st.subheader("Guardrail results")
            if guardrails["breaches"]:
                st.error(
                    f"{len(guardrails['breaches'])} guardrail breach(es): "
                    f"{', '.join(guardrails['breaches'])} -- the agent executed "
                    "a request it should have refused or escalated."
                )
            else:
                st.success(
                    f"All {guardrails['total']} guardrail tasks handled correctly."
                )

        failures = agent_eval.failure_mode_breakdown(scores)
        if failures:
            st.subheader("Failure modes")
            st.caption(
                "Clustered failures usually point at one systemic cause "
                "rather than many unrelated bugs."
            )
            st.bar_chart(pd.Series(failures, name="count"))

        st.subheader("Per-task results")
        frame = pd.DataFrame([s.to_dict() for s in scores])
        frame["failure_modes"] = frame["failure_modes"].apply(
            lambda modes: ", ".join(modes) if modes else "-"
        )
        st.dataframe(frame, width="stretch", hide_index=True)

        st.download_button(
            "Download results CSV",
            data=frame.to_csv(index=False),
            file_name=f"agent_eval_{run.version}.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Tab 2 -- Response evaluation
# --------------------------------------------------------------------------

with tab_response:
    st.subheader("Response-level evaluation")
    st.markdown(
        "Scores single question/answer pairs across five quality dimensions "
        "using Gemini as an AI judge."
    )

    dataset = llm_eval.load_dataset()
    st.caption(f"{len(dataset)} records available in the dataset.")

    sample_size = st.slider(
        "Sample size",
        min_value=1,
        max_value=min(config.MAX_SAMPLE_SIZE, max(len(dataset), 1)),
        value=min(config.DEFAULT_SAMPLE_SIZE, len(dataset)),
        help="Each record costs one judge API call.",
    )
    st.caption(
        f"Estimated cost: ${sample_size * config.COST_PER_JUDGE_CALL_USD:.4f}"
    )

    if st.button(
        "Run response evaluation",
        type="primary",
        key="run_response",
        disabled=not judge_ready,
    ):
        bar = st.progress(0.0, text="Evaluating responses...")

        def _progress(done: int, total: int) -> None:
            bar.progress(done / total, text=f"Evaluated {done}/{total}")

        run_r, results = llm_eval.run_response_eval(
            version=version,
            sample_size=sample_size,
            progress=_progress,
        )
        bar.empty()

        st.session_state["response_run"] = run_r
        st.session_state["response_results"] = results
        st.success(f"Run saved: {run_r.run_id}")

    run_r = st.session_state.get("response_run")
    results = st.session_state.get("response_results")

    if run_r and results:
        summary = run_r.summary

        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Records", summary.get("total_records", 0))
        c2.metric("Avg quality", summary.get("avg_quality_score", 0))
        c3.metric("Avg truthfulness", summary.get("avg_truthfulness", 0))
        c4.metric(
            "Avg hallucination",
            summary.get("avg_hallucination", 0),
            delta_color="inverse",
        )
        c5.metric("High hallucination", summary.get("high_hallucination_count", 0))

        frame = pd.DataFrame(results)

        flagged = llm_eval.responses_needing_review(results)
        if flagged:
            st.warning(
                f"{len(flagged)} response(s) fall below the quality threshold "
                "or were flagged for human review."
            )

        if "root_cause" in frame:
            col_rc, col_rec = st.columns(2)
            with col_rc:
                st.subheader("Root cause distribution")
                st.bar_chart(frame["root_cause"].value_counts())
            with col_rec:
                st.subheader("Recommendation distribution")
                st.bar_chart(frame["recommendation"].value_counts())

        st.subheader("Per-response results")
        st.dataframe(frame, width="stretch", hide_index=True)

        st.download_button(
            "Download results CSV",
            data=frame.to_csv(index=False),
            file_name=f"response_eval_{run_r.version}.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Tab 3 -- Regression
# --------------------------------------------------------------------------

with tab_regression:
    st.subheader("Regression check")
    st.markdown(
        "Compares two stored runs and flags metrics that moved backwards by "
        "more than the configured tolerance. This is the same logic the CI "
        "pipeline uses to fail a build."
    )

    eval_type = st.radio(
        "Evaluation type", ["trace", "response"], horizontal=True
    )
    candidates = storage.list_runs(eval_type)

    if len(candidates) < 2:
        st.info(
            f"At least two stored {eval_type} runs are needed. "
            f"Currently stored: {len(candidates)}."
        )
    else:
        labels = {
            f"{r.version}  --  {r.created_at}  ({r.run_id[-6:]})": r
            for r in candidates
        }
        keys = list(labels)

        col_base, col_curr = st.columns(2)
        baseline_key = col_base.selectbox("Baseline (older)", keys, index=1)
        current_key = col_curr.selectbox("Current (newer)", keys, index=0)

        if st.button("Compare runs", type="primary"):
            report = regression.compare_runs(
                labels[baseline_key], labels[current_key]
            )

            if report.has_regression:
                st.error(
                    f"{len(report.regressions)} regression(s) detected -- "
                    "this would fail the CI gate."
                )
            else:
                st.success("No regressions detected -- CI gate would pass.")

            st.dataframe(
                pd.DataFrame([d.to_dict() for d in report.deltas]),
                width="stretch",
                hide_index=True,
            )
            st.code(report.format_report(), language="text")