"""JSON and Markdown report writers for scored evaluation sets."""

from __future__ import annotations

import json
from typing import Dict, Sequence

from .models import CaseScore


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
        lines.append("| Metric | Mean | Min | Max | n |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for name, stats in summary.items():
            lines.append(
                f"| `{name}` | {stats['mean']:.3f} | {stats['min']:.3f} "
                f"| {stats['max']:.3f} | {stats['n']} |"
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
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path
