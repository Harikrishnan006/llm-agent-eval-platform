# LLM & Agent Evaluation Platform

Evaluation harness for LLM applications and multi-step AI agents — with
deterministic agent metrics, LLM-as-a-judge scoring, and version-to-version
regression detection that gates CI.

**Stack:** Python · Streamlit · Google Gemini · GitHub Actions

---

## Why this exists

Most LLM evaluation looks at the final output: was the answer good? That misses
how agents actually fail in production.

An agent can produce a correct answer through a broken process — a lucky guess
after calling the wrong tools. It can fail despite reasoning perfectly, because
one tool call had a malformed argument. It can loop for fourteen steps where
three would do, quietly tripling cost and latency. And it can cheerfully
execute a destructive request it should have refused.

None of that is visible from the output alone. You have to inspect the trace.

This platform evaluates at both levels, and splits metrics by how they are
computed:

| | Deterministic | LLM-as-a-judge |
|---|---|---|
| Computed from | the trace itself | a model call |
| Cost | zero | per-call |
| Reproducible | yes, exactly | approximately |
| Used to gate CI | yes | no |
| Measures | task success, tool correctness, tool hallucination, step efficiency, guardrail compliance | reasoning quality, groundedness |

The CI gate depends only on deterministic metrics. A build check that fails
because an API call timed out is worse than no check at all.

---

## Metrics

### Agent (trace-level)

| Metric | What it catches |
|---|---|
| `task_success` | Did the run achieve the intended outcome? |
| `tool_correctness` | Right tools, right order — partial credit via sequence matching |
| `tool_hallucination` | Agent called a tool that does not exist |
| `step_efficiency` | Optimal steps ÷ actual steps — catches wandering |
| `escalation_correct` | Refused when it should refuse; didn't over-escalate routine work |
| `cost_usd` / `latency_ms` | Per-run economics |
| `reasoning_quality` * | Were the steps logical and necessary? |
| `groundedness` * | Is the answer supported by what tools actually returned? |

\* judge-based, optional

### Response-level

`relevance` · `completeness` · `truthfulness` · `hallucination` · `safety`
→ combined into a composite `quality_score` (hallucination inverted).

Responses below the quality threshold, or flagged by the judge, are routed to
human review — automated scoring handles volume, humans handle the ambiguous
tail.

---

## Golden task set

Twelve tasks with known-correct outcomes across four categories:

- **knowledge_retrieval** — single-tool lookups
- **multi_step_workflow** — ordered multi-tool sequences
- **edge_case** — no valid result exists, escalation is correct
- **guardrail** — destructive, bulk, or privacy-sensitive requests the agent
  must refuse

Guardrail tasks are reported separately, because one breach matters more than a
small dip in an average. An agent that bulk-deletes customer records has failed
in a way that aggregate success rate hides.

---

## Regression detection

Every run is persisted with a version label. Comparing two runs produces a
direction-aware report — a rise in `avg_hallucination` is a regression, a rise
in `task_success_rate` is an improvement — with per-metric tolerances defined in
`src/config.py`.

```
Regression check: v1 -> v2
==========================
[!] task_success_rate          0.833 -> 0.75          (-0.083)  REGRESSION
[=] tool_correctness           0.625 -> 0.597         (-0.028)
[!] tool_hallucination_rate    0.083 -> 0.167         (+0.084)  REGRESSION
[!] escalation_accuracy        0.917 -> 0.833         (-0.084)  REGRESSION
[=] avg_steps                  5 -> 5.08              (+0.08)

FAILED: 3 regression(s) -- task_success_rate, tool_hallucination_rate, escalation_accuracy
```

`run_eval.py` exits non-zero on regression, so the GitHub Actions workflow fails
the build. A degraded agent version cannot merge silently.

---

## Quick start

```bash
git clone <repo-url>
cd llm-agent-eval-platform
pip install -r requirements.txt

# Optional — only needed for judge-based scoring
cp .env.example .env   # then add your GEMINI_API_KEY

streamlit run app.py
```

The agent evaluation tab works with no API key at all.

### CLI

```bash
# establish a baseline
python scripts/run_eval.py --type trace --version v1 --no-compare

# later run — fails (exit 1) if quality regressed
python scripts/run_eval.py --type trace --version v2

# compare against a specific version
python scripts/run_eval.py --type trace --version v3 --baseline v1

# response-level with the judge
python scripts/run_eval.py --type response --version v2 --sample-size 20 --judge
```

### Tests

```bash
python -m pytest tests/ -v
```

21 tests covering the metrics layer. All run without an API key.

---

## Project structure

```
├── app.py                      Streamlit UI (3 tabs)
├── src/
│   ├── models.py               Data models — traces, tasks, scores, runs
│   ├── metrics.py              Deterministic metrics (no LLM)
│   ├── judge.py                LLM-as-a-judge, response + trace
│   ├── llm_eval.py             Response-level orchestration
│   ├── agent_eval.py           Trace-level orchestration
│   ├── storage.py              Run persistence (JSON)
│   ├── regression.py           Baseline comparison + gating
│   └── config.py               Thresholds and tolerances
├── data/
│   ├── golden_tasks.json       12 tasks incl. 4 guardrail cases
│   ├── sample_traces.json      Example traces incl. seeded failures
│   └── runs/                   Persisted evaluation runs
├── scripts/run_eval.py         CLI runner for CI
├── tests/test_metrics.py       Unit tests
└── .github/workflows/eval.yml  CI gate
```

---

## Design notes

**Why sequence matching for tool correctness?** Exact match is pass/fail and
throws away information. An agent that called `[search, route]` when
`[search, validate, route]` was expected is closer to correct than one that
called `[delete]`. Partial credit makes regressions visible earlier.

**Why is task success keyword-based rather than judge-based?** Because it gates
CI. Keyword coverage is crude but reproducible; the judge separately assesses
groundedness where nuance actually matters.

**Why do fallbacks return neutral scores instead of raising?** A single
malformed judge response should not abort a 100-record batch. Failures are
recorded as `Evaluation Error` and surface in the root-cause distribution.

**Why plain JSON for storage?** No database to stand up, and run artefacts are
diffable and committable.

---

## Author

**Harikrishnan Venkatesan**
[LinkedIn](https://www.linkedin.com/in/harikrishnan-venkatesan-8946a3215) ·
[GitHub](https://github.com/Harikrishnan006) ·
[Hugging Face](https://huggingface.co/Harikrishnan006)
