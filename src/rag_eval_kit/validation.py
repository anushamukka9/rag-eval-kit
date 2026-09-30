"""Eval-set validation: catch format problems before a scoring run.

``validate_eval_set`` reads a JSONL file and returns a list of human-readable
issue strings (``"<path>:<line>: <message>"``). An empty list means the file is
clean. The CLI exposes this as ``rag-eval-kit validate``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _check_row(row: Any, seen_ids: set) -> List[str]:
    """Return issue strings for one decoded JSONL row (no line prefix)."""
    issues: List[str] = []
    if not isinstance(row, dict):
        return ["row is not a JSON object"]
    for field in ("question", "contexts", "answer"):
        if field not in row:
            issues.append(f"missing required field {field!r}")
    question = row.get("question")
    if "question" in row and (not isinstance(question, str) or not question.strip()):
        issues.append("question is empty")
    contexts = row.get("contexts")
    if "contexts" in row:
        if not isinstance(contexts, list):
            issues.append("contexts must be a list of strings")
        elif not contexts:
            issues.append("contexts is empty: no retrieval to evaluate")
        else:
            for i, chunk in enumerate(contexts):
                if not isinstance(chunk, str) or not chunk.strip():
                    issues.append(f"contexts[{i}] is empty")
    answer = row.get("answer")
    if "answer" in row and not isinstance(answer, str):
        issues.append("answer must be a string")
    elif isinstance(answer, str) and not answer.strip():
        issues.append("answer is empty: faithfulness and utilization will score 0")
    expected = row.get("expected")
    if expected is not None and (not isinstance(expected, str) or not expected.strip()):
        issues.append("expected is present but empty")
    case_id = row.get("id")
    if case_id is not None:
        if not isinstance(case_id, str) or not case_id.strip():
            issues.append("id is present but empty")
        elif case_id in seen_ids:
            issues.append(f"duplicate id {case_id!r}")
        else:
            seen_ids.add(case_id)
    return issues


def validate_cases(rows: List[Dict[str, Any]]) -> List[str]:
    """Validate already-decoded JSONL rows; returns issue strings with row numbers."""
    issues: List[str] = []
    seen_ids: set = set()
    for n, row in enumerate(rows, start=1):
        for issue in _check_row(row, seen_ids):
            issues.append(f"row {n}: {issue}")
    return issues


def validate_eval_set(path: str) -> List[str]:
    """Validate a JSONL eval set file.

    Returns a list of ``"<path>:<line>: <message>"`` issue strings; empty when
    the file is clean. Raises ``OSError`` if the file cannot be read.
    """
    issues: List[str] = []
    seen_ids: set = set()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(f"{path}:{lineno}: invalid JSON: {exc}")
                continue
            for issue in _check_row(row, seen_ids):
                issues.append(f"{path}:{lineno}: {issue}")
    return issues
