"""rag-eval-kit: deterministic, dependency-free evaluation for RAG pipelines.

Scores question/context/answer triples with reference-free metrics
(faithfulness, context precision/recall, answer relevancy, retrieval ranking,
context utilization, answer completeness), plus a pluggable metric registry,
a CLI, eval-set validation, and JSON/Markdown report writers.
"""

from .metrics import (
    answer_completeness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_utilization,
    faithfulness,
    list_metrics,
    register_metric,
    retrieval_ndcg,
)
from .models import CaseScore, EvalCase, MetricResult
from .scorer import aggregate_scores, load_jsonl, score_case, score_dataset
from .validation import validate_cases, validate_eval_set

__all__ = [
    "EvalCase",
    "CaseScore",
    "MetricResult",
    "faithfulness",
    "context_precision",
    "context_recall",
    "context_utilization",
    "answer_relevancy",
    "answer_completeness",
    "retrieval_ndcg",
    "register_metric",
    "list_metrics",
    "score_case",
    "score_dataset",
    "aggregate_scores",
    "load_jsonl",
    "validate_cases",
    "validate_eval_set",
]

__version__ = "0.2.0"
__author__ = "Anusha Mukka"
