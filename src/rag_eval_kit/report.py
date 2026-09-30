"""JSON and Markdown report writers for scored evaluation sets."""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

from .models import CaseScore

# Per-case score below this flags the case in the "Needs attention" section.
ATTENTION_THRESHOLD = 0.5


def build_report_dict(
    case_scores: Sequence[CaseScore], summary: Dict[str, Dict[str, float]]
) -> Dict:
    return {
        "summary": summary,
        "n_cases": len(case_scores),
        "cases": [scored.to_dict() for scored in case_scores],
    }


def write_json_report(
    case_scores: Sequence[CaseScore],
    summary: Dict[str, Dict[str, float]],
    path: str,
) -> str:
    """Write the full scored dataset and summary as JSON. Returns *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_report_dict(case_scores, summary), fh, indent=2)
    return path


def write_markdown_report(
    case_scores: Sequence[CaseScore],
    summary: Dict[str, Dict[str, float]],
    path: str,
) -> str:
    """Write a human-readable Markdown summary report. Returns *path*."""
    lines = ["# RAG Evaluation Report", ""]
    lines.append(f"Scored **{len(case_scores)}** cases.")
    lines.append("")
    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Mean | Std | Min | Max | n |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name, stats in summary.items():
            lines.append(
                f"| `{name}` | {stats['mean']:.3f} | {stats.get('std', 0.0):.3f} "
                f"| {stats['min']:.3f} | {stats['max']:.3f} | {stats['n']} |"
            )
        lines.append("")
    lines.append("## Per-case scores")
    lines.append("")
    lines.append("| Case | " + " | ".join(f"`{m}`" for m in summary) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in summary) + " |")
    for scored in case_scores:
        cells = [f"{scored.metrics[m].score:.3f}" for m in summary]
        lines.append(f"| {scored.case_id} | " + " | ".join(cells) + " |")
    lines.append("")
    attention = _cases_needing_attention(case_scores)
    lines.append("## Needs attention")
    lines.append("")
    if attention:
        lines.append(
            f"Cases with at least one metric below {ATTENTION_THRESHOLD}:"
        )
        lines.append("")
        for case_id, failures in attention:
            failing = ", ".join(f"`{name}`={score:.3f}" for name, score in failures)
            lines.append(f"- **{case_id}**: {failing}")
    else:
        lines.append(
            f"No cases scored below {ATTENTION_THRESHOLD} on any metric."
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def _cases_needing_attention(
    case_scores: Sequence[CaseScore],
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """Cases with any metric below ATTENTION_THRESHOLD, with the failing metrics."""
    flagged: List[Tuple[str, List[Tuple[str, float]]]] = []
    for scored in case_scores:
        failures = [
            (name, result.score)
            for name, result in scored.metrics.items()
            if result.score < ATTENTION_THRESHOLD
        ]
        if failures:
            flagged.append((str(scored.case_id), failures))
    return flagged
