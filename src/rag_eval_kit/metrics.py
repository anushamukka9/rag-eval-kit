"""Reference-free RAG metrics and the pluggable metric registry.

All built-in metrics are deterministic lexical heuristics over content tokens
(stopwords removed, lowercase, alphabetic, length >= 3). They need no LLM
judge, no embeddings, and no network access, so they run anywhere — including
CI — for a few cents less than a judge-based suite.

Metric functions take an :class:`~rag_eval_kit.models.EvalCase` and return a
:class:`~rag_eval_kit.models.MetricResult` with a score in [0, 1].
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Set

from .models import EvalCase, MetricResult

MetricFn = Callable[[EvalCase], MetricResult]

_REGISTRY: Dict[str, MetricFn] = {}

# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of at by for with about into
    through during before after above below to from up down in out on off over
    under again further once here there all any both each few more most other
    some such no nor not only own same so than too very can will just should
    now is are was were be been being have has had having do does did doing
    would could ought i you he she it we they them his her its our their this
    that these those am as my me your him us what which who whom whose why how
    where because until while against between among within without along also
    may might must shall need very per via used using use get got make made
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]{2,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def content_tokens(text: str) -> Set[str]:
    """Lowercased content tokens of *text*: alphabetic, length >= 3, no stopwords."""
    return {
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }


def sentences(text: str) -> List[str]:
    """Split *text* into sentences on sentence-final punctuation."""
    parts = [s.strip() for s in _SENTENCE_RE.split(text.strip())]
    return [p for p in parts if p]


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when both are empty."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_metric(name: str, fn: MetricFn | None = None) -> MetricFn:
    """Register *fn* under *name* so it runs in scoring and appears in the CLI.

    Usable directly or as a decorator::

        register_metric("my_metric", my_fn)

        @register_metric("my_metric")
        def my_metric(case: EvalCase) -> MetricResult: ...
    """
    def _register(func: MetricFn) -> MetricFn:
        if not callable(func):
            raise TypeError("metric must be callable")
        _REGISTRY[name] = func
        return func

    if fn is not None:
        return _register(fn)
    return _register


def list_metrics() -> Dict[str, MetricFn]:
    """Return a copy of the name -> metric-function registry."""
    return dict(_REGISTRY)


def get_metric(name: str) -> MetricFn:
    """Look up a registered metric by name; raises KeyError if unknown."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown metric {name!r}; available: {sorted(_REGISTRY)}"
        ) from None


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------

# Similarity at/above which a claim counts as supported by a context chunk.
SUPPORT_THRESHOLD = 0.30
# Similarity at/above which a retrieved chunk counts as relevant to the question.
RELEVANCE_THRESHOLD = 0.15


def _best_chunk_support(claim_tokens: Set[str], chunk_token_sets: List[Set[str]]) -> float:
    return max((jaccard(claim_tokens, c) for c in chunk_token_sets), default=0.0)


@register_metric("faithfulness")
def faithfulness(case: EvalCase) -> MetricResult:
    """Fraction of answer claims (sentences) supported by at least one context chunk.

    A claim is *supported* when its best chunk-level content-token Jaccard
    similarity meets ``SUPPORT_THRESHOLD``. An empty answer scores 0.0 (there
    are no claims to be faithful with). The detail map shows, per claim, its
    best support score and verdict.
    """
    claims = sentences(case.answer)
    chunk_sets = [content_tokens(c) for c in case.contexts]
    if not claims:
        return MetricResult("faithfulness", 0.0, {"claims": [], "note": "empty answer"})
    verdicts = []
    for claim in claims:
        support = _best_chunk_support(content_tokens(claim), chunk_sets)
        verdicts.append(
            {
                "claim": claim,
                "support": round(support, 4),
                "supported": support >= SUPPORT_THRESHOLD,
            }
        )
    supported = sum(1 for v in verdicts if v["supported"])
    return MetricResult(
        "faithfulness",
        supported / len(verdicts),
        {"claims": verdicts, "supported": supported, "total": len(verdicts)},
    )


@register_metric("context_precision")
def context_precision(case: EvalCase) -> MetricResult:
    """Fraction of retrieved chunks relevant to the information need.

    Relevance is judged against the reference ``expected`` answer when one is
    provided, otherwise against the question itself. A chunk is *relevant* when
    its content-token Jaccard similarity with that reference meets
    ``RELEVANCE_THRESHOLD``. Penalizes retrieving distractors alongside useful
    chunks.
    """
    if not case.contexts:
        return MetricResult("context_precision", 0.0, {"note": "no contexts retrieved"})
    reference_tokens = content_tokens(case.expected if case.expected else case.question)
    verdicts = []
    for i, chunk in enumerate(case.contexts):
        relevance = jaccard(reference_tokens, content_tokens(chunk))
        verdicts.append(
            {
                "chunk_index": i,
                "relevance": round(relevance, 4),
                "relevant": relevance >= RELEVANCE_THRESHOLD,
            }
        )
    relevant = sum(1 for v in verdicts if v["relevant"])
    return MetricResult(
        "context_precision",
        relevant / len(verdicts),
        {"chunks": verdicts, "relevant": relevant, "total": len(verdicts)},
    )


@register_metric("context_recall")
def context_recall(case: EvalCase) -> MetricResult:
    """Fraction of the information need covered by the retrieved contexts.

    The information need is the content-token set of the reference ``expected``
    answer when provided, otherwise of the question itself. A need token is
    *covered* when it appears in any retrieved chunk. Penalizes retrieving
    chunks that miss the point of the question.
    """
    need_source = case.expected if case.expected else case.question
    need = content_tokens(need_source)
    if not need:
        return MetricResult("context_recall", 0.0, {"note": "empty information need"})
    covered = {tok for chunk in case.contexts for tok in content_tokens(chunk)} & need
    return MetricResult(
        "context_recall",
        len(covered) / len(need),
        {
            "covered": sorted(covered),
            "missing": sorted(need - covered),
            "covered_count": len(covered),
            "need_count": len(need),
        },
    )


@register_metric("answer_relevancy")
def answer_relevancy(case: EvalCase) -> MetricResult:
    """How directly the answer addresses the question.

    Content-token Jaccard similarity between the answer and the question.
    Off-topic or boilerplate-heavy answers score low; terse on-topic answers
    score high.
    """
    question_tokens = content_tokens(case.question)
    answer_tokens = content_tokens(case.answer)
    if not question_tokens or not answer_tokens:
        return MetricResult("answer_relevancy", 0.0, {"note": "empty question or answer"})
    score = jaccard(question_tokens, answer_tokens)
    return MetricResult(
        "answer_relevancy",
        score,
        {
            "question_tokens": len(question_tokens),
            "answer_tokens": len(answer_tokens),
            "shared_tokens": len(question_tokens & answer_tokens),
        },
    )
