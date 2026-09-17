"""rag-eval-kit: deterministic, dependency-free evaluation for RAG pipelines.

Scores question/context/answer triples with reference-free metrics
(faithfulness, context precision/recall, answer relevancy), plus a pluggable
metric registry, a CLI, and JSON/Markdown report writers.
"""

from .metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    list_metrics,
    register_metric,
)
from .models import CaseScore, EvalCase, MetricResult
from .scorer import aggregate_scores, load_jsonl, score_case, score_dataset

__all__ = [
    "EvalCase",
    "CaseScore",
    "MetricResult",
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "register_metric",
    "list_metrics",
    "score_case",
    "score_dataset",
    "aggregate_scores",
    "load_jsonl",
]

__version__ = "0.1.0"
__author__ = "Anusha Mukka"
