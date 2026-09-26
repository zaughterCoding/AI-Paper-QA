"""Contract tests for the frozen, chunk-independent benchmark."""
import copy
import importlib.util
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec("scripts.benchmark") is not None, "benchmark scorer is not implemented"
    from scripts import benchmark
    return benchmark


def fixture():
    corpus = {"a": "alpha beta gamma delta epsilon", "b": "other source evidence"}
    questions = [
        {"id": "yes", "answerable": True, "category": "cross_paper", "split": "test",
         "evidence_groups": [[{"source_id": "a", "start": 6, "end": 16}],
                             [{"source_id": "b", "start": 0, "end": 12}]]},
        {"id": "no", "answerable": False, "category": "unanswerable", "split": "test", "evidence_groups": []},
    ]
    return corpus, questions


def test_partial_evidence_does_not_mean_complete_answer_and_duplicates_do_not_help():
    b = api()
    corpus, questions = fixture()
    rows = [{"id": "yes", "contexts": [{"source_id": "a", "text": "alpha beta gamma"}] * 2},
            {"id": "no", "contexts": []}]
    result = b.score(questions, rows, corpus, k=5)
    assert result["retrieval"]["evidence_recall"] == 0.5
    assert result["retrieval"]["complete_evidence_rate"] == 0
    assert result["retrieval"]["mrr"] == 1
    assert result["answerability"] is None


def test_adjacent_chunks_jointly_cover_evidence_and_alternative_sources_are_valid():
    b = api()
    corpus, questions = fixture()
    rows = [{"id": "yes", "contexts": [{"source_id": "a", "text": "alpha beta"},
                                           {"source_id": "a", "text": "gamma delta"},
                                           {"source_id": "b", "text": "other source evidence"}]},
            {"id": "no", "contexts": []}]
    result = b.score(questions, rows, corpus, k=3)
    assert result["retrieval"]["complete_evidence_rate"] == 1
    assert result["retrieval"]["mrr"] == 0.5
    questions[0]["evidence_groups"] = [[{"source_id": "a", "start": 6, "end": 16},
                                        {"source_id": "b", "start": 0, "end": 12}]]
    rows[0]["contexts"] = rows[0]["contexts"][-1:]
    assert b.score(questions, rows, corpus)["retrieval"]["complete_evidence_rate"] == 1


def test_confusion_matrix_has_all_four_cells_and_wrong_answers_are_not_accuracy():
    b = api()
    corpus, questions = fixture()
    questions = [dict(questions[i % 2], id=str(i)) for i in range(4)]
    rows = [{"id": str(i), "contexts": [], "answered": i < 2,
             "answer_correct": False if i == 0 else None} for i in range(4)]
    result = b.score(questions, rows, corpus)
    assert result["answerability"]["confusion_matrix"] == {"TP": 1, "FP": 1, "TN": 1, "FN": 1}
    assert result["answerability"]["precision"] == 0.5
    assert result["generation"]["judged_answer_count"] == 1
    assert result["generation"]["answer_accuracy"] == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "fabricated", "partial_decisions", "bad_decision"])
def test_invalid_predictions_fail_instead_of_silently_changing_denominators(fault):
    b = api()
    corpus, questions = fixture()
    rows = [{"id": q["id"], "contexts": []} for q in questions]
    if fault == "missing": rows.pop()
    if fault == "duplicate": rows.append(copy.deepcopy(rows[0]))
    if fault == "unknown": rows[0]["id"] = "unknown"
    if fault == "fabricated": rows[0]["contexts"] = [{"source_id": "a", "text": "invented evidence"}]
    if fault == "partial_decisions": rows[0]["answered"] = True
    if fault == "bad_decision":
        for r in rows: r["answered"] = "false"
    with pytest.raises(ValueError): b.score(questions, rows, corpus)


def test_word_budget_cannot_be_bypassed_by_returning_whole_papers():
    b = api()
    corpus, questions = fixture()
    rows = [{"id": "yes", "contexts": [{"source_id": "a", "text": corpus["a"]},
                                           {"source_id": "b", "text": corpus["b"]}]},
            {"id": "no", "contexts": []}]
    assert b.score(questions, rows, corpus, max_words=1)["retrieval"]["evidence_recall"] == 0


def test_frozen_dataset_is_grounded_balanced_and_papers_are_ignored():
    b = api()
    questions, corpus, manifest = b.load_benchmark()
    assert len(questions) >= 100
    assert len(corpus) == 20
    assert {q["split"] for q in questions} == {"dev", "test"}
    assert {"terminology", "numeric", "semantic", "cross_section", "cross_paper", "unanswerable"} <= {q["category"] for q in questions}
    assert sum(not q["answerable"] for q in questions) >= 20
    assert {e["source_id"] for q in questions for group in q["evidence_groups"] for e in group} == set(corpus)
    assert manifest["baseline_commit"] == "5bcc07487d822df9edfe0f7f6b326af19acf1890"
    import subprocess
    paths = [str(p.relative_to(b.ROOT)) for p in (b.ROOT / "eval/corpus").glob("*.txt")]
    result = subprocess.run(["git", "check-ignore", *paths], cwd=b.ROOT, capture_output=True, text=True, check=True)
    assert len(result.stdout.splitlines()) == 20
    assert not subprocess.check_output(["git", "ls-files", "eval/corpus/*.txt"], cwd=b.ROOT, text=True).strip()


def test_paired_comparison_reports_question_level_regressions():
    b = api()
    assert hasattr(b, "compare"), "paired comparison is not implemented"
    corpus, questions = fixture()
    bad = [{"id": q["id"], "contexts": []} for q in questions]
    good = copy.deepcopy(bad)
    good[0]["contexts"] = [{"source_id": s, "text": text} for s, text in corpus.items()]
    old = b.score(questions, good, corpus)
    new = b.score(questions, bad, corpus)
    comparison = b.compare(new, old)
    assert comparison["regressions"] == ["yes"]
    assert comparison["retrieval_delta"]["complete_evidence_rate"] == -1
    old["max_context_words"] = 100
    with pytest.raises(ValueError): b.compare(new, old)


def test_missing_citation_judgments_are_not_treated_as_correct():
    b = api()
    corpus, questions = fixture()
    rows = [{"id": "yes", "contexts": [], "answered": True,
             "citation_supported": 1, "citation_count": 2,
             "claims_supported": 1, "claim_count": 3},
            {"id": "no", "contexts": [], "answered": False}]
    result = b.score(questions, rows, corpus)
    assert result["generation"].get("citation_precision") == 0.5
    assert result["generation"].get("claim_support_rate") == pytest.approx(1/3)
    rows[0]["citation_supported"] = 3
    with pytest.raises(ValueError): b.score(questions, rows, corpus)


def test_pinned_download_rejects_changed_content_before_writing(tmp_path, monkeypatch):
    assert importlib.util.find_spec("scripts.fetch_benchmark") is not None, "pinned fetcher is not implemented"
    from scripts import fetch_benchmark as fetch
    b = api()
    from types import SimpleNamespace
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(text="unexpected", raise_for_status=lambda: None)
    monkeypatch.setattr(fetch.httpx, "get", get)
    monkeypatch.setattr(fetch, "extract_text", lambda html: html)
    paper = {"source_id": "p", "file": "p.txt", "arxiv_id": "1234.56789", "version": "arXiv:1234.56789v2 [cs.CL]",
             "normalized_sha256": b.digest("expected")}
    with pytest.raises(ValueError): fetch.fetch_paper(paper, tmp_path)
    assert calls == ["https://arxiv.org/html/1234.56789v2"]
    assert not (tmp_path / "p.txt").exists()
    (tmp_path / "p.txt").write_text("existing local content", encoding="utf-8")
    with pytest.raises(ValueError): fetch.fetch_paper(paper, tmp_path)
    assert (tmp_path / "p.txt").read_text() == "existing local content"


def test_context_offsets_at_whitespace_boundaries_preserve_evidence():
    b = api()
    corpus = {"a": "alpha beta gamma"}
    assert b.context_span({"source_id": "a", "start": 5, "end": 16}, corpus, 2) == ("a", 6, 16, 2)
    assert b.context_span({"source_id": "a", "start": 5, "end": 16, "text": " beta gamma"}, corpus, 1) == ("a", 6, 10, 1)


def test_composite_questions_and_their_component_facts_do_not_cross_splits():
    b = api()
    questions, _, _ = b.load_benchmark()
    fact_splits = {}
    for q in questions:
        for group in q["evidence_groups"]:
            for e in group:
                key = (e["source_id"], e["start"], e["end"])
                prior = fact_splits.setdefault(key, q["split"])
                assert prior == q["split"], f"Evidence fact leaks across splits: {q['id']}"


def test_generation_cannot_claim_fair_comparison_after_using_extra_context():
    b = api()
    corpus, questions = fixture()
    rows = [{"id": "yes", "contexts": [{"source_id": "a", "text": corpus["a"]}], "answered": True},
            {"id": "no", "contexts": [], "answered": False}]
    with pytest.raises(ValueError, match="budget"):
        b.score(questions, rows, corpus, max_words=1)
