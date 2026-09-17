"""Tests for rag-eval-kit's metrics, registry, scorer, reports, and CLI."""

import json
import subprocess
import sys

import pytest

from rag_eval_kit import (
    EvalCase,
    aggregate_scores,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    list_metrics,
    load_jsonl,
    register_metric,
    score_case,
    score_dataset,
)
from rag_eval_kit.cli import main as cli_main
from rag_eval_kit.report import write_json_report, write_markdown_report


def good_case() -> EvalCase:
    return EvalCase(
        id="good-1",
        question="What causes photosynthesis in plants?",
        contexts=[
            "Photosynthesis in plants is driven by sunlight captured by chlorophyll in the leaves.",
            "Chlorophyll absorbs light energy and converts carbon dioxide and water into glucose.",
            "The best pizza toppings include pepperoni and mushrooms.",
        ],
        answer="Photosynthesis in plants is caused by sunlight captured by chlorophyll in the leaves. Chlorophyll converts carbon dioxide and water into glucose.",
        expected="Sunlight captured by chlorophyll drives photosynthesis in plants, converting carbon dioxide and water into glucose.",
    )


def bad_case() -> EvalCase:
    return EvalCase(
        id="bad-1",
        question="What causes photosynthesis in plants?",
        contexts=[
            "The best pizza toppings include pepperoni and mushrooms.",
            "Penguins live in Antarctica and eat fish.",
        ],
        answer="Photosynthesis is caused by moonlight and the tides, according to ancient legends.",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_faithfulness_supported_answer_scores_high():
    result = faithfulness(good_case())
    assert result.score >= 0.5
    assert result.detail["supported"] >= 1


def test_faithfulness_unsupported_answer_scores_low():
    result = faithfulness(bad_case())
    assert result.score == pytest.approx(0.0)


def test_faithfulness_empty_answer_scores_zero():
    case = good_case()
    case.answer = ""
    assert faithfulness(case).score == 0.0


def test_context_precision_penalizes_distractors():
    result = context_precision(good_case())
    # 2 of 3 chunks are about photosynthesis; the pizza chunk is a distractor.
    assert result.score == pytest.approx(2 / 3)


def test_context_precision_empty_contexts_scores_zero():
    case = good_case()
    case.contexts = []
    assert context_precision(case).score == 0.0


def test_context_recall_covers_expected_answer_tokens():
    result = context_recall(good_case())
    assert result.score > 0.5
    assert "chlorophyll" in result.detail["covered"]


def test_context_recall_falls_back_to_question_without_expected():
    case = good_case()
    case.expected = None
    result = context_recall(case)
    assert 0.0 <= result.score <= 1.0


def test_context_recall_misses_everything_scores_zero():
    assert context_recall(bad_case()).score == pytest.approx(0.0)


def test_answer_relevancy_on_topic_scores_higher_than_off_topic():
    assert answer_relevancy(good_case()).score > answer_relevancy(bad_case()).score


def test_answer_relevancy_empty_answer_scores_zero():
    case = good_case()
    case.answer = ""
    assert answer_relevancy(case).score == 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_register_metric_decorator_and_list():
    from rag_eval_kit.models import MetricResult

    @register_metric("test_constant_metric")
    def constant(case: EvalCase) -> MetricResult:
        return MetricResult("test_constant_metric", 0.42)

    assert "test_constant_metric" in list_metrics()
    scored = score_case(good_case(), metric_names=["test_constant_metric"])
    assert scored.metrics["test_constant_metric"].score == pytest.approx(0.42)


def test_unknown_metric_raises_key_error():
    with pytest.raises(KeyError):
        score_case(good_case(), metric_names=["no_such_metric"])


# ---------------------------------------------------------------------------
# Scorer + reports
# ---------------------------------------------------------------------------


def test_score_dataset_and_aggregate(tmp_path):
    cases = [good_case(), bad_case()]
    scored = score_dataset(cases)
    assert len(scored) == 2
    summary = aggregate_scores(scored)
    # Custom metrics registered by other tests may also appear; the built-ins
    # must always be present.
    assert {"faithfulness", "context_precision", "context_recall", "answer_relevancy"} <= set(summary)
    for stats in summary.values():
        assert stats["n"] == 2
        assert stats["min"] <= stats["mean"] <= stats["max"]


def test_load_jsonl_round_trip(tmp_path):
    path = tmp_path / "eval.jsonl"
    rows = [good_case().to_dict(), bad_case().to_dict()]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    loaded = load_jsonl(str(path))
    assert len(loaded) == 2
    assert loaded[0].id == "good-1"
    assert loaded[1].question.startswith("What causes")


def test_load_jsonl_rejects_bad_row(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "q"}\n')
    with pytest.raises(ValueError):
        load_jsonl(str(path))


def test_json_and_markdown_reports(tmp_path):
    scored = score_dataset([good_case(), bad_case()])
    summary = aggregate_scores(scored)
    json_path = write_json_report(scored, summary, str(tmp_path / "report.json"))
    md_path = write_markdown_report(scored, summary, str(tmp_path / "report.md"))
    data = json.loads(open(json_path).read())
    assert data["n_cases"] == 2
    assert "faithfulness" in data["summary"]
    md = open(md_path).read()
    assert "# RAG Evaluation Report" in md
    assert "faithfulness" in md
    assert "good-1" in md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_eval_set(tmp_path) -> str:
    path = tmp_path / "eval.jsonl"
    rows = [good_case().to_dict(), bad_case().to_dict()]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_cli_metrics_lists_builtins(capsys):
    assert cli_main(["metrics"]) == 0
    out = capsys.readouterr().out
    for name in ("faithfulness", "context_precision", "context_recall", "answer_relevancy"):
        assert name in out


def test_cli_score_writes_json_report(tmp_path, capsys):
    input_path = _write_eval_set(tmp_path)
    out_path = str(tmp_path / "out.json")
    assert cli_main(["score", input_path, "--output", out_path]) == 0
    data = json.loads(open(out_path).read())
    assert data["n_cases"] == 2


def test_cli_score_markdown_and_metric_filter(tmp_path):
    input_path = _write_eval_set(tmp_path)
    out_path = str(tmp_path / "out.md")
    assert (
        cli_main(
            ["score", input_path, "--output", out_path, "--format", "markdown",
             "--metrics", "faithfulness"]
        )
        == 0
    )
    md = open(out_path).read()
    assert "faithfulness" in md
    assert "context_precision" not in md


def test_cli_fail_under_exits_nonzero(tmp_path, capsys):
    input_path = _write_eval_set(tmp_path)
    rc = cli_main(
        ["score", input_path, "--output", str(tmp_path / "o.json"), "--fail-under", "0.99"]
    )
    assert rc == 1


def test_cli_missing_file_returns_error(tmp_path, capsys):
    rc = cli_main(["score", str(tmp_path / "nope.jsonl")])
    assert rc == 2
    assert "error" in capsys.readouterr().err


def test_console_script_entrypoint(tmp_path):
    """The installed ``rag-eval-kit`` script behaves like the CLI module."""
    input_path = _write_eval_set(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "rag_eval_kit", "metrics"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "faithfulness" in proc.stdout
    assert input_path  # keeps the fixture in play
