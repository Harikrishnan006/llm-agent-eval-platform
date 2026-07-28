# 🎯 LLM & Agent Evaluation Platform

> Evaluation harness for LLM applications and multi-step AI agents — deterministic agent metrics, LLM-as-a-Judge scoring, and version-to-version regression detection that gates CI.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**[▶ Live Demo](https://llm-agent-eval-platform-8bmmeaapstqguqecxhqgcj.streamlit.app/)** — the Agent Evaluation tab runs with no API key required.

---

## 📸 Screenshots

### Agent evaluation — catching a guardrail breach
<!-- Screenshot: Agent Evaluation tab after clicking "Run agent evaluation".
     Capture the metric row + the red guardrail breach box + failure modes chart. -->
![Agent evaluation results](docs/screenshot-agent-eval.png)

*Deterministic metrics across 12 golden tasks. The agent that bulk-emailed instead of refusing is flagged as a guardrail breach; the invented `sla_lookup` tool is caught as a tool hallucination.*

### Regression detection — failing a build on quality drop
<!-- Screenshot: Regression tab after comparing a degraded version against baseline.
     Capture the red "regressions detected" banner + the delta table. -->
![Regression report](docs/screenshot-regression.png)

*Direction-aware thresholds. A rise in `tool_hallucination_rate` is a regression; a rise in `task_success_rate` is an improvement.*

---

## 🤔 Why this exists

Most LLM evaluation looks at the final output: was the answer good? That misses how agents actually fail in production.

An agent can produce a correct answer through a broken process — a lucky guess after calling the wrong tools. It can fail despite reasoning perfectly, because one tool call had a malformed argument. It can loop for fourteen steps where three would do, quietly tripling cost and latency. And it can cheerfully execute a destructive request it should have refused.

None of that is visible from the output alone. You have to inspect the trace.

This platform evaluates at both levels, and splits metrics by **how they are computed**:

| | Deterministic | LLM-as-a-Judge |
|---|---|---|
| Computed from | the trace itself | a model call |
| Cost | zero | per-call |
| Reproducible | exactly | approximately |
| Gates CI | ✅ yes | ❌ no |
| Measures | task success, tool correctness, tool hallucination, step efficiency, guardrail compliance | reasoning quality, groundedness |

The CI gate depends only on deterministic metrics. A build check that fails because an API call timed out is worse than no check at all.

---

## 📊 Metrics

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

`relevance` · `completeness` · `truthfulness` · `hallucination` · `safety` → combined into a composite `quality_score` (hallucination inverted).

Responses below the quality threshold, or flagged by the judge, are routed to human review — automated scoring handles volume, humans handle the ambiguous tail.

---

## 🎯 Golden task set

Twelve tasks with known-correct outcomes across four categories:

| Category | Tasks | Tests |
|---|---|---|
| `knowledge_retrieval` | 4 | Single-tool lookups |
| `multi_step_workflow` | 4 | Ordered multi-tool sequences |
| `edge_case` | 1 | No valid result exists — escalation is correct |
| `guardrail` | 3 | Destructive / bulk / privacy-sensitive requests the agent must refuse |

Guardrail tasks are reported separately, because one breach matters more than a small dip in an average. An agent that bulk-deletes customer records has failed in a way that aggregate success rate hides.

---

## 🔁 Regression detection

Every run is persisted with a version label. Comparing two runs produces a direction-aware report with per-metric tolerances defined in `src/config.py`.

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

`run_eval.py` exits non-zero on regression, so the GitHub Actions workflow fails the build. A degraded agent version cannot merge silently.

---

## 🏗 Project structure

```
llm-agent-eval-platform/
├── app.py                          # Streamlit UI — 3 tabs, explicit run buttons
├── requirements.txt
├── .env.example                    # GEMINI_API_KEY template (judge only)
│
├── src/
│   ├── models.py          (303)    # Data models — AgentTrace, TraceStep, GoldenTask,
│   │                               #   TraceScore, ResponseScore, EvalRun + enums
│   ├── metrics.py         (251)    # Deterministic metrics — NO LLM. Tool correctness
│   │                               #   via sequence matching, hallucination detection,
│   │                               #   step efficiency, guardrail compliance
│   ├── judge.py           (236)    # LLM-as-a-Judge — response + trace scoring,
│   │                               #   constrained enums, fence stripping, safe fallback
│   ├── agent_eval.py      (188)    # Trace-level orchestration + failure-mode clustering
│   ├── llm_eval.py        (176)    # Response-level orchestration + human-review routing
│   ├── regression.py      (166)    # Baseline comparison, direction-aware thresholds
│   ├── storage.py          (94)    # Versioned run persistence (JSON)
│   └── config.py           (78)    # Thresholds, tolerances, API key resolution
│
├── data/
│   ├── golden_tasks.json           # 12 tasks incl. 4 guardrail refusal cases
│   ├── sample_traces.json          # Example traces with seeded failure modes
│   └── runs/baseline.json          # Committed baseline — what CI compares against
│
├── scripts/
│   └── run_eval.py                 # CLI runner — exits non-zero on regression
│
├── tests/
│   └── test_metrics.py             # 21 unit tests, no API key required
│
└── .github/workflows/eval.yml      # CI gate — tests + evaluation on every PR
```

---

## ⚡ Quick start

```bash
git clone https://github.com/Harikrishnan006/llm-agent-eval-platform
cd llm-agent-eval-platform

pip install -r requirements.txt

# Optional — only needed for judge-based scoring
cp .env.example .env    # then add your GEMINI_API_KEY

streamlit run app.py
```

The **Agent Evaluation** tab works with no API key at all.

---

## 🖥 CLI

```bash
# establish a baseline
python scripts/run_eval.py --type trace --version v1 --no-compare

# later run — exits 1 if quality regressed
python scripts/run_eval.py --type trace --version v2

# compare against a specific version
python scripts/run_eval.py --type trace --version v3 --baseline v1

# response-level with the judge
python scripts/run_eval.py --type response --version v2 --sample-size 20 --judge
```

| Flag | Purpose |
|---|---|
| `--type` | `trace` (agent runs) or `response` (Q/A pairs) |
| `--version` | Version label stored with the run |
| `--baseline` | Version to compare against (default: most recent) |
| `--no-compare` | Skip regression check — for establishing a first baseline |
| `--judge` | Enable LLM-as-a-Judge scoring |
| `--json-out` | Write the regression report as JSON |

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

21 tests covering the metrics layer — tool sequence matching, hallucination detection, step efficiency, escalation logic, composite scoring, and aggregation. **All run without an API key**, which is exactly why the deterministic split was worth building.

---

## 🔧 Design notes

**Why sequence matching for tool correctness?** Exact match is pass/fail and throws away information. An agent that called `[search, route]` when `[search, validate, route]` was expected is closer to correct than one that called `[delete]`. Partial credit makes regressions visible earlier.

**Why is task success keyword-based rather than judge-based?** Because it gates CI. Keyword coverage is crude but reproducible; the judge separately assesses groundedness where nuance actually matters.

**Why do fallbacks return neutral scores instead of raising?** A single malformed judge response should not abort a 100-record batch. Failures are recorded as `Evaluation Error` and surface in the root-cause distribution.

**Why plain JSON for storage?** No database to stand up, and run artefacts are diffable and committable — which is what makes the baseline a reviewable part of the repo.

---

## 🚀 Roadmap

- [ ] Emit traces from a live CrewAI / LangGraph agent instead of authored fixtures
- [ ] Langfuse tracing integration for automatic trace capture
- [ ] MCP server exposing the harness as a tool
- [ ] Per-metric cost attribution across model providers

---

## ⚠️ Notes

- Gemini free tier allows **20 requests/day** — the Response Evaluation tab will show `Evaluation Error` once exhausted. The Agent Evaluation tab is unaffected (no API calls).
- Sample traces are hand-authored to demonstrate each failure mode. Building the harness against known-correct outcomes is the intended approach; wiring it to a live agent is the next step (see Roadmap).

---

## 👨‍💻 Author

**Harikrishnan Venkatesan** — Applied AI Engineer

[LinkedIn](https://www.linkedin.com/in/harikrishnan-venkatesan-8946a3215) · [GitHub](https://github.com/Harikrishnan006) · [Hugging Face](https://huggingface.co/Harikrishnan006)
