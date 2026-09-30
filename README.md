# rag-eval-kit

A deterministic, dependency-free evaluation toolkit for RAG pipelines. Score
question/context/answer triples with reference-free metrics — no LLM judge, no
embeddings, no network — and get JSON or Markdown summary reports from a single
CLI command.

## Metrics

| Metric | What it measures | Needs `expected`? |
| --- | --- | --- |
| `faithfulness` | Fraction of answer sentences supported by at least one retrieved chunk | no |
| `context_precision` | Fraction of retrieved chunks relevant to the question (or reference answer, when provided) | no |
| `context_recall` | Fraction of the information need covered by the retrieved chunks | optional (improves it) |
| `context_utilization` | Fraction of retrieved chunks the answer actually draws on | no |
| `retrieval_ndcg` | Ranking quality: are the most relevant chunks retrieved first? | optional (improves it) |
| `answer_relevancy` | How directly the answer addresses the question | no |
| `answer_completeness` | Fraction of the reference answer's content present in the answer | yes |

All seven are lexical heuristics over content tokens (stopwords removed): cheap,
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
# Validate the eval set format before scoring
rag-eval-kit validate examples/sample_eval_set.jsonl

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
summary = aggregate_scores(scored)   # per-metric mean / std / min / max
write_markdown_report(scored, summary, "report.md")
```

The Markdown report includes a per-case score table and a "Needs attention"
section listing every case with a metric below 0.5, so weak spots surface
without digging through JSON.

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
├── models.py     # EvalCase, MetricResult, CaseScore dataclasses
├── metrics.py    # token utilities, built-in metrics, metric registry
├── scorer.py     # JSONL loading, per-case scoring, aggregate statistics
├── validation.py # eval-set format checks (`rag-eval-kit validate`)
├── report.py     # JSON and Markdown report writers
└── cli.py        # `rag-eval-kit` console script (`score`, `validate`, `metrics`)
```

The pipeline is `JSONL → validate → EvalCase → MetricResult per (case, metric)
→ aggregate statistics → report`. Metrics are pure functions registered by
name, so the scorer, CLI, and reports all pick up custom metrics with no extra
wiring.

## Development

```bash
pip install -e . && pip install pytest
pytest -q
```

## License

MIT — Copyright (c) 2026 [Anusha Mukka](https://anushamukka.com).
