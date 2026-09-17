# rag-eval-kit usage guide

This guide covers the library API, the metric registry, the CLI, and the
report formats. For the 60-second tour, see `examples/quickstart.py`.

## The evaluation unit

Everything is built on one record type:

```python
from rag_eval_kit import EvalCase

case = EvalCase(
    question="What causes photosynthesis in plants?",
    contexts=[
        "Photosynthesis in plants is driven by sunlight captured by chlorophyll.",
        "The best pizza toppings include pepperoni and mushrooms.",
    ],
    answer="Photosynthesis is driven by sunlight captured by chlorophyll.",
    expected="Sunlight captured by chlorophyll drives photosynthesis.",  # optional
    id="q1",  # optional
)
```

`contexts` should be the retrieved chunks in retrieval order. `expected` is an
optional reference answer used by context recall; without it, recall falls back
to the question text.

## Built-in metrics

| Metric | What it measures | Needs `expected`? |
| --- | --- | --- |
| `faithfulness` | Fraction of answer sentences supported by at least one context chunk | no |
| `context_precision` | Fraction of retrieved chunks relevant to the question | no (uses `expected` as the relevance reference when present) |
| `context_recall` | Fraction of the information need covered by the retrieved chunks | optional (improves it) |
| `answer_relevancy` | How directly the answer addresses the question | no |

All metrics are deterministic lexical heuristics over content tokens (stopwords
removed). They are reference-free: they need no ground-truth answers, no LLM
judge, and no network. That makes them cheap, reproducible, and CI-friendly —
a good first pass before (or alongside) expensive judge-based evaluation.

Thresholds live in `rag_eval_kit.metrics` as `SUPPORT_THRESHOLD` (default 0.30)
and `RELEVANCE_THRESHOLD` (default 0.15); adjust them to your corpus if the
defaults are too strict or too lenient.

## Scoring a dataset

Eval sets are JSONL files, one `EvalCase` per line:

```json
{"id": "q1", "question": "...", "contexts": ["..."], "answer": "...", "expected": "..."}
```

```python
from rag_eval_kit import load_jsonl, score_dataset, aggregate_scores
from rag_eval_kit.report import write_json_report, write_markdown_report

cases = load_jsonl("eval.jsonl")
scored = score_dataset(cases)                    # or score_dataset(cases, ["faithfulness"])
summary = aggregate_scores(scored)               # per-metric mean/min/max

write_json_report(scored, summary, "report.json")
write_markdown_report(scored, summary, "report.md")
```

## Custom metrics

Register any function that takes an `EvalCase` and returns a `MetricResult`
(score clamped to [0, 1]):

```python
from rag_eval_kit import EvalCase, MetricResult, register_metric, score_case

@register_metric("answer_length_ok")
def answer_length_ok(case: EvalCase) -> MetricResult:
    words = len(case.answer.split())
    return MetricResult("answer_length_ok", 1.0 if 10 <= words <= 200 else 0.0,
                        {"words": words})

result = score_case(case, ["answer_length_ok", "faithfulness"])
```

Registered metrics automatically run in `score_dataset`, appear in
`rag-eval-kit metrics`, and are selectable with the CLI `--metrics` flag.

## CLI

```bash
# List metrics
rag-eval-kit metrics

# Score an eval set (JSON report)
rag-eval-kit score eval.jsonl --output report.json

# Markdown report, subset of metrics
rag-eval-kit score eval.jsonl --output report.md --format markdown \
    --metrics faithfulness,answer_relevancy

# Both formats at once
rag-eval-kit score eval.jsonl --output report --format both

# Gate a CI pipeline: fail if any metric mean drops below 0.7
rag-eval-kit score eval.jsonl --output report.json --fail-under 0.7
```

Exit codes: `0` success, `1` a `--fail-under` threshold tripped, `2` usage or
input errors.

## Report formats

- **JSON** (`report.json`): `summary` (per-metric mean/min/max), `n_cases`,
  and per-case scores with each metric's `detail` payload — machine-readable
  for dashboards and regression tracking.
- **Markdown** (`report.md`): summary table plus a per-case score table —
  readable in a PR or pasted into a doc.
