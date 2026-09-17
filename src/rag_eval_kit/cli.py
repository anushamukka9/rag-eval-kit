"""Command-line interface for rag-eval-kit.

Subcommands:
    score    Score a JSONL eval set and write a summary report.
    metrics  List the registered metrics.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .metrics import list_metrics
from .report import write_json_report, write_markdown_report
from .scorer import aggregate_scores, load_jsonl, score_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval-kit",
        description="Evaluate RAG pipelines on question/context/answer triples.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score a JSONL eval set and write a report.")
    p_score.add_argument("input", help="Path to the JSONL eval set.")
    p_score.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names to run (default: all registered).",
    )
    p_score.add_argument(
        "--output",
        default="report.json",
        help="Report output path (default: report.json).",
    )
    p_score.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="json",
        help="Report format (default: json).",
    )
    p_score.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit non-zero if any metric's mean score is below this threshold.",
    )

    sub.add_parser("metrics", help="List the registered metrics.")
    return parser


def _cmd_score(args: argparse.Namespace) -> int:
    metric_names = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics
        else None
    )
    try:
        cases = load_jsonl(args.input)
        case_scores = score_dataset(cases, metric_names)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = aggregate_scores(case_scores)

    written = []
    if args.format in ("json", "both"):
        path = args.output if args.output.endswith(".json") else args.output + ".json"
        write_json_report(case_scores, summary, path)
        written.append(path)
    if args.format in ("markdown", "both"):
        base = args.output
        path = base if base.endswith(".md") else base.rsplit(".", 1)[0] + ".md"
        write_markdown_report(case_scores, summary, path)
        written.append(path)

    print(f"scored {len(case_scores)} cases")
    for name, stats in summary.items():
        print(f"  {name}: mean={stats['mean']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}")
    print("wrote " + ", ".join(written))

    if args.fail_under is not None:
        failing = [n for n, s in summary.items() if s["mean"] < args.fail_under]
        if failing:
            print(
                f"FAIL: mean score below {args.fail_under}: {', '.join(failing)}",
                file=sys.stderr,
            )
            return 1
    return 0


def _cmd_metrics(_args: argparse.Namespace) -> int:
    for name in sorted(list_metrics()):
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "score":
        return _cmd_score(args)
    if args.command == "metrics":
        return _cmd_metrics(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
