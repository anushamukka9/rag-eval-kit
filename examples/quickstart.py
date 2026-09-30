"""Quickstart example: validate and score a tiny eval set, then write reports.

Run from the repo root:

    python examples/quickstart.py
"""

import json
import subprocess
import sys
from pathlib import Path

from rag_eval_kit import (
    aggregate_scores,
    list_metrics,
    load_jsonl,
    score_dataset,
    validate_eval_set,
)
from rag_eval_kit.report import write_json_report, write_markdown_report


def main() -> None:
    root = Path(__file__).resolve().parent
    eval_set = root / "sample_eval_set.jsonl"

    # 0. Validate the eval set before scoring it.
    issues = validate_eval_set(str(eval_set))
    if issues:
        print("eval set has issues:")
        for issue in issues:
            print(" ", issue)
        raise SystemExit(1)
    print("eval set is clean\n")

    # 1. Python API: load, score, aggregate.
    cases = load_jsonl(str(eval_set))
    scored = score_dataset(cases)
    summary = aggregate_scores(scored)

    print(f"Registered metrics: {sorted(list_metrics())}\n")
    for name, stats in summary.items():
        print(
            f"{name:20s} mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
            f"min={stats['min']:.3f}  max={stats['max']:.3f}"
        )

    # 2. Per-case breakdown: which metrics fell below 0.5, and why.
    print("\nPer-case breakdown:")
    for case in scored:
        weak = [n for n, r in case.metrics.items() if r.score < 0.5]
        status = "OK" if not weak else f"weak on {', '.join(weak)}"
        print(f"  {case.case_id}: {status}")

    # 3. Write both report formats.
    write_json_report(scored, summary, str(root / "quickstart_report.json"))
    write_markdown_report(scored, summary, str(root / "quickstart_report.md"))
    print("\nwrote examples/quickstart_report.json and examples/quickstart_report.md")

    # 4. Same thing through the CLI (mirrors what CI would do).
    print("\n--- CLI run ---")
    subprocess.run(
        [sys.executable, "-m", "rag_eval_kit", "validate", str(eval_set)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_eval_kit",
            "score",
            str(eval_set),
            "--output",
            str(root / "quickstart_cli.json"),
            "--format",
            "both",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
