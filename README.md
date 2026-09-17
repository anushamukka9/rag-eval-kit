# rag-eval-kit

A deterministic, dependency-free evaluation toolkit for RAG pipelines. Score
question/context/answer triples with reference-free metrics — no LLM judge, no
embeddings, no network — and get JSON or Markdown summary reports from a single
CLI command.

## Metrics

| Metric | What it measures |
| --- | --- |
| `faithfulness` | Fraction of answer sentences supported by at least one retrieved chunk |
| `context_precision` | Fraction of retrieved chunks relevant to the question (or reference answer, when provided) |
| `context_recall` | Fraction of the information need covered by the retrieved chunks |
| `answer_relevancy` | How directly the answer addresses the question |

All four are lexical heuristics over content tokens (stopwords removed): cheap,
reproducible, and CI-friendly. Bring your own metrics via the pluggable
registry — registered metrics run in scoring and the CLI automatically.

## Install

```bash
pip install git+https://github.com/anushamukka9/rag-eval-kit.git
# or from source:
git clone https://github.com/anushamukka9/rag-eval-kit.git
cd rag-eval-kit && pip install -e .
```

Requires Python 3.9+. No runtime dependencies.

## Quickstart

```bash
# List available metrics
rag-eval-kit metrics

# Score an eval set (see examples/sample_eval_set.jsonl for the format)
rag-eval-kit score examples/sample_eval_set.jsonl --output report --format both
```

```python
from rag_eval_kit import EvalCase, aggregate_scores, score_dataset, load_jsonl
from rag_eval_kit.report import write_markdown_report

cases = load_jsonl("eval.jsonl")
scored = score_dataset(cases)
summary = aggregate_scores(scored)
write_markdown_report(scored, summary, "report.md")
```

Custom metric in five lines:

```python
from rag_eval_kit import EvalCase, MetricResult, register_metric

@register_metric("answer_length_ok")
def answer_length_ok(case: EvalCase) -> MetricResult:
    words = len(case.answer.split())
    return MetricResult("answer_length_ok", 1.0 if 10 <= words <= 200 else 0.0)
```

More in [`docs/usage.md`](docs/usage.md) and the runnable
[`examples/quickstart.py`](examples/quickstart.py).

## Architecture

```
src/rag_eval_kit/
├── models.py    # EvalCase, MetricResult, CaseScore dataclasses
├── metrics.py   # token utilities, built-in metrics, metric registry
├── scorer.py    # JSONL loading, per-case scoring, aggregate statistics
├── report.py    # JSON and Markdown report writers
└── cli.py       # `rag-eval-kit` console script (`score`, `metrics`)
```

The pipeline is `JSONL → EvalCase → MetricResult per (case, metric) → aggregate
statistics → report`. Metrics are pure functions registered by name, so the
scorer, CLI, and reports all pick up custom metrics with no extra wiring.

## Development

```bash
pip install -e . && pip install pytest
pytest -q
```

## License

MIT — Copyright (c) 2026 [Anusha Mukka](https://anushamukka.com).
