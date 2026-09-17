"""Dataset loading, per-case scoring, and aggregate statistics."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence

from .metrics import get_metric
from .models import CaseScore, EvalCase, MetricResult


def load_jsonl(path: str) -> List[EvalCase]:
    """Load evaluation cases from a JSONL file.

    Each line must be a JSON object with ``question``, ``contexts`` (list),
    and ``answer``; optional ``expected`` and ``id``. Blank lines are skipped.
    """
    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                case = EvalCase.from_dict(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if case.id is None:
                case.id = f"case-{lineno}"
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no evaluation cases found")
    return cases


def score_case(case: EvalCase, metric_names: Optional[Sequence[str]] = None) -> CaseScore:
    """Run the selected metrics (or all registered ones) on one case."""
    from .metrics import list_metrics

    names = list(metric_names) if metric_names else sorted(list_metrics())
    results: Dict[str, MetricResult] = {}
    for name in names:
        results[name] = get_metric(name)(case)
    return CaseScore(case_id=case.id, metrics=results)


def score_dataset(
    cases: Iterable[EvalCase], metric_names: Optional[Sequence[str]] = None
) -> List[CaseScore]:
    """Score every case in *cases* and return the per-case results."""
    return [score_case(case, metric_names) for case in cases]


def aggregate_scores(case_scores: Sequence[CaseScore]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-case metric scores into mean/min/max over the dataset."""
    if not case_scores:
        return {}
    per_metric: Dict[str, List[float]] = {}
    for scored in case_scores:
        for name, result in scored.metrics.items():
            per_metric.setdefault(name, []).append(result.score)
    summary: Dict[str, Dict[str, float]] = {}
    for name in sorted(per_metric):
        values = per_metric[name]
        summary[name] = {
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "n": len(values),
        }
    return summary
