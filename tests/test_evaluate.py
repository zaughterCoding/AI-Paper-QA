"""Tests for the evaluation script.

Two layers, tested differently.

The metrics are pure functions over a list of results, and their mistakes have no
symptom: an off-by-one in the rank comparison turns 22/25 into 25/25, and a mean that
silently counts empty retrievals as 0.0 produces a plausible lower number. Neither shows
up when the script is run by hand, so the inputs are built to order and the outputs pinned
exactly.

The loader reads a file written by hand, so it is tested against files built in ``tmp_path``
with one defect each, and every rejection is asserted to name the line.

``eval/questions.jsonl`` itself is checked too, against ``eval/corpus/``: that is the only
guard against the corpus being refetched under an evaluation set whose labels no longer
describe it. Titles and terms that stop matching would otherwise leave every question
still loading and still scoring, with the expected source simply never found again.
"""

import json
import uuid
from pathlib import Path

import pytest

from app.repositories.chunks import RetrievedChunk
from scripts.evaluate import (
    EvalQuestion,
    EvalResult,
    collect_top_scores,
    compute_hit_rate,
    compute_rank1_hit_rate,
    describe_rank1_misses,
    load_eval_questions,
    summarize,
    summarize_top_scores,
)
from scripts.fetch_corpus import PAPERS

REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
CORPUS_DIR = REPO_ROOT / "eval" / "corpus"
SOURCES_PATH = CORPUS_DIR / "sources.json"

ATTENTION = "Attention Is All You Need"
BERT = "BERT: Pre-training of Deep Bidirectional Transformers"


def uuid_of(*parts: object) -> uuid.UUID:
    """A deterministic UUID built from the arguments, so tests stay reproducible."""
    return uuid.uuid5(uuid.NAMESPACE_URL, "-".join(str(part) for part in parts))


def chunk(title: str, score: float) -> RetrievedChunk:
    """A RetrievedChunk carrying only what the metrics read: title and score.

    The identity fields are filled with fixed placeholders. Nothing in this module looks
    at them, and generating real ones would suggest they matter here.
    """
    return RetrievedChunk(
        chunk_id=uuid_of(title, score),
        document_id=uuid_of(title, 0),
        title=title,
        chunk_index=0,
        text="text",
        score=score,
    )


def question(text: str = "What is self-attention?", source: str = ATTENTION) -> EvalQuestion:
    return EvalQuestion(question=text, expected_terms=["self-attention"], expected_source=source)


def result(
    chunks: list[RetrievedChunk], text: str = "What is self-attention?", source: str = ATTENTION
) -> EvalResult:
    return EvalResult(question=question(text, source), retrieved=chunks)


# --- loading: the happy path --------------------------------------------------


def write(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_load_reads_every_field(tmp_path) -> None:
    path = write(
        tmp_path,
        {"question": "Q1", "expected_terms": ["a", "b"], "expected_source": ATTENTION},
        {"question": "Q2", "expected_terms": ["c"], "expected_source": BERT},
    )

    questions = load_eval_questions(path)

    assert questions == [
        EvalQuestion(question="Q1", expected_terms=["a", "b"], expected_source=ATTENTION),
        EvalQuestion(question="Q2", expected_terms=["c"], expected_source=BERT),
    ]


def test_load_keeps_the_file_order(tmp_path) -> None:
    """Order is what lets two runs be compared line by line, and what makes the
    ``rank1_misses`` list read in the same sequence as the file."""
    path = write(
        tmp_path,
        {"question": "first", "expected_terms": ["a"], "expected_source": ATTENTION},
        {"question": "second", "expected_terms": ["b"], "expected_source": BERT},
    )

    assert [q.question for q in load_eval_questions(path)] == ["first", "second"]


def test_load_skips_blank_lines(tmp_path) -> None:
    """A trailing newline is not a malformed question, and neither is a gap between
    groups of questions."""
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"question": "Q1", "expected_terms": ["a"], "expected_source": "S"}\n'
        "\n"
        "   \n"
        '{"question": "Q2", "expected_terms": ["b"], "expected_source": "S"}\n',
        encoding="utf-8",
    )

    assert len(load_eval_questions(path)) == 2


def test_load_handles_crlf_line_endings(tmp_path) -> None:
    """The file is committed with LF and checked out with CRLF on Windows.

    Splitting on "\\n" alone would leave a carriage return inside the last field of every
    line, where it is invisible until a title comparison fails.
    """
    path = tmp_path / "questions.jsonl"
    path.write_bytes(
        b'{"question": "Q1", "expected_terms": ["a"], "expected_source": "S"}\r\n'
        b'{"question": "Q2", "expected_terms": ["b"], "expected_source": "S"}\r\n'
    )

    questions = load_eval_questions(path)

    assert [q.question for q in questions] == ["Q1", "Q2"]
    assert questions[0].expected_source == "S"


# --- loading: what must be rejected -------------------------------------------


def test_load_rejects_a_missing_file(tmp_path) -> None:
    with pytest.raises(OSError):
        load_eval_questions(tmp_path / "absent.jsonl")


def test_load_rejects_malformed_json_and_names_the_line(tmp_path) -> None:
    """A line number is what turns "the file is broken" into something fixable."""
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"question": "Q1", "expected_terms": ["a"], "expected_source": "S"}\n'
        "{not json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r":2:"):
        load_eval_questions(path)


def test_load_rejects_a_json_array_line(tmp_path) -> None:
    """Valid JSON is not enough: each line has to be an object.

    A file accidentally written as an array of arrays parses line by line and would
    otherwise fail later with a confusing error about subscripts.
    """
    path = tmp_path / "questions.jsonl"
    path.write_text('["Q1", ["a"], "S"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON object"):
        load_eval_questions(path)


@pytest.mark.parametrize("missing", ["question", "expected_terms", "expected_source"])
def test_load_rejects_a_missing_field(tmp_path, missing: str) -> None:
    row = {"question": "Q1", "expected_terms": ["a"], "expected_source": "S"}
    del row[missing]
    path = write(tmp_path, row)

    with pytest.raises(ValueError, match=missing):
        load_eval_questions(path)


def test_load_rejects_an_unknown_field(tmp_path) -> None:
    """A field the loader does not know is rejected rather than ignored.

    Ignoring it is how a label the author believed was in use goes unscored for a whole
    task without anyone noticing -- which is exactly what happened to ``expected_terms``,
    present in every line and read by nothing.
    """
    path = write(
        tmp_path,
        {
            "question": "Q1",
            "expected_terms": ["a"],
            "expected_sources": "S",
            "expected_source": "S",
        },
    )

    with pytest.raises(ValueError, match="expected_sources"):
        load_eval_questions(path)


def test_load_rejects_terms_as_a_bare_string(tmp_path) -> None:
    """The check that matters most here.

    ``"expected_terms": "self-attention"`` is valid JSON and iterates happily into
    thirteen one-character terms, so nothing downstream ever raises.
    """
    path = write(
        tmp_path, {"question": "Q1", "expected_terms": "self-attention", "expected_source": "S"}
    )

    with pytest.raises(ValueError, match="non-empty list"):
        load_eval_questions(path)


def test_load_rejects_an_empty_terms_list(tmp_path) -> None:
    path = write(tmp_path, {"question": "Q1", "expected_terms": [], "expected_source": "S"})

    with pytest.raises(ValueError, match="non-empty list"):
        load_eval_questions(path)


def test_load_rejects_a_non_string_term(tmp_path) -> None:
    path = write(tmp_path, {"question": "Q1", "expected_terms": ["a", 3], "expected_source": "S"})

    with pytest.raises(ValueError, match="non-empty string"):
        load_eval_questions(path)


@pytest.mark.parametrize("field", ["question", "expected_source"])
def test_load_rejects_an_empty_string_field(tmp_path, field: str) -> None:
    row = {"question": "Q1", "expected_terms": ["a"], "expected_source": "S"}
    row[field] = "   "
    path = write(tmp_path, row)

    with pytest.raises(ValueError, match="non-empty string"):
        load_eval_questions(path)


def test_load_rejects_a_duplicate_question(tmp_path) -> None:
    """A repeated question is scored twice, silently doubling its weight in every rate
    here. Nothing downstream can detect that."""
    path = write(
        tmp_path,
        {"question": "Q1", "expected_terms": ["a"], "expected_source": "S"},
        {"question": "Q1", "expected_terms": ["b"], "expected_source": "S"},
    )

    with pytest.raises(ValueError, match="duplicate question"):
        load_eval_questions(path)


def test_load_rejects_a_file_with_no_questions(tmp_path) -> None:
    """An empty file is a broken benchmark, not a benchmark that scores nothing.

    Returning an empty list would move the failure to the metrics, where it surfaces as a
    division guard and reads like a programming error rather than a wrong path.
    """
    path = tmp_path / "questions.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no questions found"):
        load_eval_questions(path)


# --- hit rate -----------------------------------------------------------------


def test_hit_rate_counts_every_hit() -> None:
    results = [
        result([chunk(ATTENTION, 0.6)]),
        result([chunk(BERT, 0.5), chunk(ATTENTION, 0.4)], text="Q2", source=ATTENTION),
    ]

    assert compute_hit_rate(results) == 1.0


def test_hit_rate_counts_none_when_the_source_never_appears() -> None:
    results = [result([chunk(BERT, 0.5)]), result([chunk(BERT, 0.4)], text="Q2")]

    assert compute_hit_rate(results) == 0.0


def test_hit_rate_is_a_fraction() -> None:
    results = [result([chunk(ATTENTION, 0.6)]), result([chunk(BERT, 0.5)], text="Q2")]

    assert compute_hit_rate(results) == 0.5


def test_hit_rate_ignores_rank() -> None:
    """The loose rate exists to be passed; it must not quietly demand rank 1.

    Pinned because the strict rate is computed from the same field, and tightening this
    one into it would make both metrics the same number while leaving the output looking
    unchanged.
    """
    results = [result([chunk(BERT, 0.6), chunk(ATTENTION, 0.5)])]

    assert compute_hit_rate(results) == 1.0


def test_hit_rate_counts_a_repeated_source_once() -> None:
    """Two chunks from the expected document are one hit, not two.

    Papers contribute several overlapping chunks, so this is the normal case rather than
    an edge one.
    """
    results = [
        result([chunk(ATTENTION, 0.6), chunk(ATTENTION, 0.5), chunk(ATTENTION, 0.4)])
    ]

    assert compute_hit_rate(results) == 1.0


def test_hit_rate_rejects_an_empty_result_list() -> None:
    """Scoring zero questions returns no number rather than 0.0.

    0.0 would be read as "every question failed" when the truth is "no question was run".
    """
    with pytest.raises(ValueError, match="no results"):
        compute_hit_rate([])


# --- rank-1 hit rate ----------------------------------------------------------


def test_rank1_hit_rate_counts_only_first_place() -> None:
    results = [
        result([chunk(ATTENTION, 0.6)]),
        result([chunk(BERT, 0.5), chunk(ATTENTION, 0.4)], text="Q2"),
    ]

    assert compute_rank1_hit_rate(results) == 0.5


def test_rank1_hit_rate_is_stricter_than_the_loose_rate() -> None:
    """The whole reason both exist, on one example.

    The same results score 1.0 loose and 0.0 strict. On the real corpus the gap is smaller
    but the shape is the same: the loose rate calls a run perfect while a question's top
    result is the wrong document.
    """
    results = [result([chunk(BERT, 0.6), chunk(ATTENTION, 0.5)])]

    assert compute_hit_rate(results) == 1.0
    assert compute_rank1_hit_rate(results) == 0.0


def test_rank1_hit_rate_counts_a_result_with_nothing_retrieved_as_a_miss() -> None:
    results = [result([]), result([chunk(ATTENTION, 0.5)], text="Q2")]

    assert compute_rank1_hit_rate(results) == 0.5


def test_rank1_hit_rate_rejects_an_empty_result_list() -> None:
    with pytest.raises(ValueError, match="no results"):
        compute_rank1_hit_rate([])


# --- scores -------------------------------------------------------------------


def test_collect_top_scores_takes_the_best_chunk_of_each_question() -> None:
    results = [
        result([chunk(ATTENTION, 0.62), chunk(BERT, 0.11)]),
        result([chunk(BERT, 0.45)], text="Q2"),
    ]

    assert collect_top_scores(results) == [0.62, 0.45]


def test_collect_top_scores_skips_questions_that_retrieved_nothing() -> None:
    """Not 0.0. A question with no results has no score, and folding in a zero would
    invent a measurement no run can produce -- one that would then set the minimum."""
    results = [result([]), result([chunk(ATTENTION, 0.5)], text="Q2")]

    assert collect_top_scores(results) == [0.5]


def test_summarize_top_scores_reports_the_distribution_and_its_size() -> None:
    results = [
        result([chunk(ATTENTION, 0.62)]),
        result([chunk(ATTENTION, 0.30)], text="Q2"),
        result([chunk(ATTENTION, 0.46)], text="Q3"),
    ]

    assert summarize_top_scores(results) == {
        "count": 3,
        "min": 0.30,
        "median": 0.46,
        "max": 0.62,
    }


def test_summarize_top_scores_reveals_a_skipped_question_through_its_count() -> None:
    """count is the only field that says a question was left out of the statistics.

    Without it the mean and minimum would describe fewer questions than question_count
    with nothing in the output to show it, which is how a partial run gets read as a
    complete one.
    """
    results = [result([]), result([chunk(ATTENTION, 0.5)], text="Q2")]

    assert summarize_top_scores(results)["count"] == 1


def test_summarize_top_scores_rejects_a_list_with_no_scores() -> None:
    with pytest.raises(ValueError, match="no scores"):
        summarize_top_scores([result([])])


def test_summary_keeps_four_decimals() -> None:
    """Two decimals is not enough here.

    The number a similarity threshold has to be set below is the minimum, and the real
    distribution's minimum is 0.3591 -- which rounding to 0.36 would blur into the same
    value as several neighbours.
    """
    results = [result([chunk(ATTENTION, 0.3591)]), result([chunk(ATTENTION, 0.6683)], text="Q2")]

    summary = summarize(results)

    assert summary["top_score"]["min"] == 0.3591
    assert summary["avg_top_score"] == 0.5137


# --- the summary shape --------------------------------------------------------


def test_summary_reports_the_planned_keys() -> None:
    """The three keys the plan specifies must be present under those exact names, since
    the output is meant to be read by whatever runs the script."""
    summary = summarize([result([chunk(ATTENTION, 0.5)])])

    assert summary["question_count"] == 1
    assert summary["retrieval_hit_rate"] == 1.0
    assert summary["avg_top_score"] == 0.5


def test_summary_reports_both_rates_side_by_side() -> None:
    """Both rates in one output, because either alone misleads: the loose one saturates,
    the strict one hides that the source was found at all."""
    results = [result([chunk(BERT, 0.6), chunk(ATTENTION, 0.5)])]

    summary = summarize(results)

    assert summary["retrieval_hit_rate"] == 1.0
    assert summary["retrieval_rank1_hit_rate"] == 0.0


def test_summary_has_exactly_the_documented_keys() -> None:
    """Pins the output contract.

    The summary is the script's whole product and is meant to be diffed between runs, so a
    key appearing or disappearing should be a deliberate edit with a failing test beside
    it rather than a side effect of a refactor.
    """
    summary = summarize([result([chunk(ATTENTION, 0.5)])])

    assert set(summary) == {
        "question_count",
        "retrieval_hit_rate",
        "retrieval_rank1_hit_rate",
        "avg_top_score",
        "top_score",
        "rank1_misses",
    }


# --- naming the failures ------------------------------------------------------


def test_describe_rank1_misses_names_the_question_and_what_beat_it() -> None:
    results = [result([chunk(BERT, 0.6), chunk(ATTENTION, 0.5)])]

    assert describe_rank1_misses(results) == [
        {
            "question": "What is self-attention?",
            "expected_source": ATTENTION,
            "expected_rank": 2,
            "ranked_first": BERT,
        }
    ]


def test_describe_rank1_misses_reports_a_source_that_never_came_back() -> None:
    """expected_rank is null here, which is how a complete miss is told apart from a
    result that was merely ranked second."""
    results = [result([chunk(BERT, 0.6)])]

    assert describe_rank1_misses(results)[0]["expected_rank"] is None


def test_describe_rank1_misses_includes_a_question_with_nothing_retrieved() -> None:
    """It missed rank 1 by the widest margin there is, so leaving it out would make this
    list shorter than the failures the rates count."""
    misses = describe_rank1_misses([result([])])

    assert misses == [
        {
            "question": "What is self-attention?",
            "expected_source": ATTENTION,
            "expected_rank": None,
            "ranked_first": None,
        }
    ]


def test_describe_rank1_misses_is_empty_when_everything_ranks_first() -> None:
    results = [result([chunk(ATTENTION, 0.6)]), result([chunk(BERT, 0.5)], text="Q2", source=BERT)]

    assert describe_rank1_misses(results) == []


def test_describe_rank1_misses_length_matches_the_strict_rate() -> None:
    """The list and the rate must agree: a list shorter than the failures would send
    someone to fix three questions while four are broken."""
    results = [
        result([chunk(ATTENTION, 0.6)]),
        result([chunk(BERT, 0.5)], text="Q2"),
        result([chunk(BERT, 0.4), chunk(ATTENTION, 0.3)], text="Q3"),
        result([], text="Q4"),
    ]

    misses = len(describe_rank1_misses(results))

    assert misses == 3
    assert compute_rank1_hit_rate(results) == (len(results) - misses) / len(results)


# --- the real evaluation set --------------------------------------------------


@pytest.fixture(scope="module")
def real_questions() -> list[EvalQuestion]:
    return load_eval_questions(QUESTIONS_PATH)


def test_the_shipped_evaluation_set_loads(real_questions) -> None:
    assert len(real_questions) >= 20


def test_the_shipped_evaluation_set_has_no_duplicate_questions(real_questions) -> None:
    questions = [q.question for q in real_questions]

    assert len(set(questions)) == len(questions)


def test_every_expected_source_is_a_corpus_title(real_questions) -> None:
    """The guard against the corpus changing under a fixed evaluation set.

    ``expected_source`` is compared against ``documents.title`` as a plain string, with no
    foreign key behind it. Refetching the corpus under a new title would leave every
    question loading and scoring normally, with its expected source simply never found
    again -- 25 perfect questions turned into 25 unexplained failures, with nothing
    pointing at the corpus as the cause.
    """
    titles = {paper.title for paper in PAPERS}

    unknown = sorted({q.expected_source for q in real_questions} - titles)

    assert unknown == []


def test_every_expected_term_appears_in_the_document_it_is_labelled_with(real_questions) -> None:
    """Same guard, for the terms.

    They are not scored yet, so a term that no longer occurs costs nothing today -- and
    would cost a silent re-reading of every paper on the day an answer-quality metric
    starts using them.
    """
    files = {paper.title: CORPUS_DIR / f"{paper.slug}.txt" for paper in PAPERS}

    missing = []
    for question in real_questions:
        text = files[question.expected_source].read_text(encoding="utf-8")
        missing += [
            (question.expected_source, term)
            for term in question.expected_terms
            if term not in text
        ]

    assert missing == []


def test_the_shipped_evaluation_set_labels_the_corpus_it_ships_with(real_questions) -> None:
    """sources.json and the .txt files must describe the same five papers.

    Read rather than assumed: the titles in this module and in the loader both come from
    it, so a manifest listing a file that is not there would make the guard above pass
    against a corpus that no longer exists.
    """
    manifest = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))

    assert {entry["title"] for entry in manifest["papers"]} == {paper.title for paper in PAPERS}
    assert all((CORPUS_DIR / entry["file"]).exists() for entry in manifest["papers"])
