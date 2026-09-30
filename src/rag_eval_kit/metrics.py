"""Reference-free RAG metrics and the pluggable metric registry.

All built-in metrics are deterministic lexical heuristics over content tokens
(stopwords removed, lowercase, alphabetic, length >= 3). They need no LLM
judge, no embeddings, and no network access, so they run anywhere — including
CI — for a few cents less than a judge-based suite.

Metric functions take an :class:`~rag_eval_kit.models.EvalCase` and return a
:class:`~rag_eval_kit.models.MetricResult` with a score in [0, 1].
"""

from __future__ import annotations

import math
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


# Chunk relevance grades for retrieval_ndcg: (similarity floor, grade).
_RELEVANCE_GRADES = ((0.50, 3), (0.30, 2), (RELEVANCE_THRESHOLD, 1))


@register_metric("retrieval_ndcg")
def retrieval_ndcg(case: EvalCase) -> MetricResult:
    """NDCG of the retrieved chunk ranking: are the best chunks ranked first?

    Each chunk gets a relevance grade (0-3) from its content-token Jaccard
    similarity with the reference ``expected`` answer (or the question when no
    reference is given). The score is DCG over the retrieved order divided by
    the DCG of the ideal ordering. Scores 1.0 when nothing is retrievable to
    mis-rank; 0.0 when no chunks were retrieved at all.
    """
    if not case.contexts:
        return MetricResult("retrieval_ndcg", 0.0, {"note": "no contexts retrieved"})
    reference_tokens = content_tokens(case.expected if case.expected else case.question)
    grades = []
    for chunk in case.contexts:
        sim = jaccard(reference_tokens, content_tokens(chunk))
        grade = 0
        for floor, g in _RELEVANCE_GRADES:
            if sim >= floor:
                grade = g
                break
        grades.append({"similarity": round(sim, 4), "grade": grade})

    def _dcg(ordered_grades):
        return sum((2.0 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ordered_grades))

    retrieved = [g["grade"] for g in grades]
    dcg = _dcg(retrieved)
    idcg = _dcg(sorted(retrieved, reverse=True))
    if idcg == 0:
        return MetricResult(
            "retrieval_ndcg", 1.0, {"grades": grades, "note": "no relevant chunks to rank"}
        )
    return MetricResult(
        "retrieval_ndcg",
        dcg / idcg,
        {"grades": grades, "dcg": round(dcg, 4), "idcg": round(idcg, 4)},
    )


@register_metric("context_utilization")
def context_utilization(case: EvalCase) -> MetricResult:
    """Fraction of retrieved chunks that support at least one answer claim.

    Complements ``context_precision`` (chunks relevant to the question) by
    asking whether the generator actually used what was retrieved. A chunk
    *supports* a claim when their content-token Jaccard similarity meets
    ``SUPPORT_THRESHOLD``. Low utilization with high precision means the
    generator is ignoring good retrieval.
    """
    if not case.contexts:
        return MetricResult("context_utilization", 0.0, {"note": "no contexts retrieved"})
    claims = sentences(case.answer)
    if not claims:
        return MetricResult("context_utilization", 0.0, {"note": "empty answer"})
    chunk_sets = [content_tokens(c) for c in case.contexts]
    verdicts = []
    for i, chunk_set in enumerate(chunk_sets):
        supporting = sum(
            1
            for claim in claims
            if jaccard(content_tokens(claim), chunk_set) >= SUPPORT_THRESHOLD
        )
        verdicts.append(
            {"chunk_index": i, "used": supporting > 0, "supporting_claims": supporting}
        )
    used = sum(1 for v in verdicts if v["used"])
    return MetricResult(
        "context_utilization",
        used / len(verdicts),
        {"chunks": verdicts, "used": used, "total": len(verdicts)},
    )


@register_metric("answer_completeness")
def answer_completeness(case: EvalCase) -> MetricResult:
    """Fraction of the reference answer's content covered by the generated answer.

    Requires the ``expected`` reference answer; scores 0.0 with a note when it
    is missing. A need token counts as covered when it appears in the answer's
    content tokens. Penalizes answers that address the question but omit key
    facts from the reference.
    """
    if not case.expected:
        return MetricResult(
            "answer_completeness", 0.0, {"note": "no expected answer provided"}
        )
    need = content_tokens(case.expected)
    if not need:
        return MetricResult(
            "answer_completeness", 0.0, {"note": "empty expected answer"}
        )
    have = content_tokens(case.answer)
    covered = need & have
    return MetricResult(
        "answer_completeness",
        len(covered) / len(need),
        {
            "covered": sorted(covered),
            "missing": sorted(need - covered),
            "covered_count": len(covered),
            "need_count": len(need),
        },
    )
