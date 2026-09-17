"""Core data models: evaluation cases and scored results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalCase:
    """One evaluation example: a question, its retrieved contexts, and the answer.

    Attributes:
        question: The user's question.
        contexts: Retrieved context chunks (in retrieval order).
        answer: The generated answer to evaluate.
        expected: Optional reference answer, used by context recall when present.
        id: Optional stable identifier for reporting.
    """

    question: str
    contexts: List[str]
    answer: str
    expected: Optional[str] = None
    id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalCase":
        """Build an EvalCase from a plain dict (e.g. one JSONL row)."""
        try:
            return cls(
                question=str(data["question"]),
                contexts=[str(c) for c in data["contexts"]],
                answer=str(data["answer"]),
                expected=None if data.get("expected") is None else str(data["expected"]),
                id=None if data.get("id") is None else str(data["id"]),
            )
        except KeyError as exc:
            raise ValueError(f"missing required field in eval row: {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "contexts": self.contexts,
            "answer": self.answer,
            "expected": self.expected,
        }


@dataclass
class MetricResult:
    """The outcome of running one metric on one case."""

    name: str
    score: float  # 0.0 .. 1.0
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp defensively so custom metrics can't poison aggregates.
        self.score = max(0.0, min(1.0, float(self.score)))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 4), "detail": self.detail}


@dataclass
class CaseScore:
    """All metric scores for a single evaluation case."""

    case_id: Optional[str]
    metrics: Dict[str, MetricResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "metrics": {name: res.to_dict() for name, res in self.metrics.items()},
        }
