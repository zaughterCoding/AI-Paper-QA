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

That check runs on both kinds of question and in opposite directions. A positive's terms
must occur in the document it names; a negative's must occur nowhere in the corpus at all,
because a term that does occur is proof that the corpus holds an answer the question claims
it does not. The corpus files are the only place either claim can be tested, which makes
this the one thing standing behind the negative set -- the loader can check that a negative
is well-formed, but only the corpus can say whether it is true.
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
    describe_negatives_above,
    describe_rank1_misses,
    load_eval_questions,
    negative_results,
    positive_results,
    separating_gap,
    summarize,
    summarize_top_scores,
    threshold_separates,
    threshold_sweep,
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


def question(text: str = "What is self-attention?", source: str | None = ATTENTION) -> EvalQuestion:
    return EvalQuestion(question=text, expected_terms=["self-attention"], expected_source=source)


def result(
    chunks: list[RetrievedChunk],
    text: str = "What is self-attention?",
    source: str | None = ATTENTION,
) -> EvalResult:
    return EvalResult(question=question(text, source), retrieved=chunks)


def positive_questions(questions: list[EvalQuestion]) -> list[EvalQuestion]:
    """The positive questions out of a loaded file.

    Separate from ``positive_results``, which splits ``EvalResult`` and is what the metrics
    use. A loaded file holds ``EvalQuestion``, so passing one to that helper reaches for
    ``.question`` on a str.
    """
    return [question for question in questions if not question.is_negative]


def negative(
    chunks: list[RetrievedChunk], text: str = "What learning rate schedule is recommended?"
) -> EvalResult:
    """A question the corpus is not expected to answer, with chunks retrieved anyway.

    The chunks matter: a negative that retrieved nothing is the outcome the sweep hopes
    for, and one that retrieved something is what the sweep exists to price. Tests that
    want the first pass an empty list to ``result`` with ``source=None`` directly.
    """
    return EvalResult(question=question(text, source=None), retrieved=chunks)


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


# --- loading: the negative marker ----------------------------------------------


def test_load_reads_a_negative_question(tmp_path) -> None:
    """A null expected_source is a question, not a missing value.

    It survives the round trip as None, which is the whole of what marks the question as
    one the corpus is not expected to answer.
    """
    path = write(
        tmp_path,
        {"question": "Q1", "expected_terms": ["a"], "expected_source": ATTENTION},
        {"question": "Q2", "expected_terms": ["b"], "expected_source": None},
    )

    questions = load_eval_questions(path)

    assert questions[0].is_negative is False
    assert questions[1].is_negative is True
    assert questions[1].expected_source is None


def test_load_rejects_an_empty_string_source_and_says_what_null_is_for(tmp_path) -> None:
    """The empty string is not a way of saying "no document", and not a title either.

    Let through, it would equal no document's title and the question would score as a
    permanent, unexplained miss. The message has to name null: a writer who typed "" was
    reaching for the negative marker and needs to be told where it actually is.
    """
    path = write(tmp_path, {"question": "Q1", "expected_terms": ["a"], "expected_source": ""})

    with pytest.raises(ValueError, match="null") as error:
        load_eval_questions(path)

    assert "non-empty" in str(error.value)


def test_load_rejects_a_file_where_every_question_is_negative(tmp_path) -> None:
    """Every rate here is a fraction of the positives, so such a file could only fail
    later, further from the cause and with a worse message."""
    path = write(
        tmp_path,
        {"question": "Q1", "expected_terms": ["a"], "expected_source": None},
        {"question": "Q2", "expected_terms": ["b"], "expected_source": None},
    )

    with pytest.raises(ValueError, match="no positive questions"):
        load_eval_questions(path)


def test_load_rejects_a_term_that_is_both_required_and_forbidden(tmp_path) -> None:
    """The two uses of expected_terms contradict each other, and no corpus can satisfy
    both.

    Naming both questions is the point. Failing here instead of in the corpus check is the
    difference between "some term is missing from one file and present in another" and
    "these two questions disagree, and one of them is wrong".
    """
    path = write(
        tmp_path,
        {"question": "asks for it", "expected_terms": ["warmup"], "expected_source": ATTENTION},
        {"question": "rules it out", "expected_terms": ["warmup"], "expected_source": None},
    )

    with pytest.raises(ValueError, match="warmup") as error:
        load_eval_questions(path)

    assert "asks for it" in str(error.value)
    assert "rules it out" in str(error.value)


def test_load_says_how_to_write_a_negative_when_it_rejects_an_unknown_field(tmp_path) -> None:
    """The rejection has to name the way through, not only the way out.

    A null expected_source is the least obvious part of the format, and the most natural
    wrong move -- inventing an "is_negative": true field -- lands on this rejection. A
    message saying only that the field is unknown leaves the writer to guess.
    """
    path = write(
        tmp_path,
        {
            "question": "Q1",
            "expected_terms": ["a"],
            "expected_source": None,
            "is_negative": True,
        },
    )

    with pytest.raises(ValueError, match="is_negative") as error:
        load_eval_questions(path)

    assert "null" in str(error.value)


# --- what a negative question means --------------------------------------------


def test_a_negatives_terms_are_forbidden_and_a_positives_are_required() -> None:
    """The two properties exist as a pair so a caller cannot reach for the wrong side.

    There is deliberately no single attribute holding "the terms", because its meaning
    depends on the kind of question and using it on the wrong kind would invert the
    corpus check silently.
    """
    assert question().required_terms == ["self-attention"]
    assert question().forbidden_terms == []

    assert question(source=None).required_terms == []
    assert question(source=None).forbidden_terms == ["self-attention"]


def test_expected_rank_is_an_error_for_a_negative_question() -> None:
    """Not None, which is what the comparison would return if it were allowed to run.

    A chunk title is never None, so the loop finds no match and returns the same None it
    returns for a positive whose source was genuinely not retrieved. That is a silent wrong
    answer in place of a missing one: it makes every negative a retrieval failure, in the
    rate and in the list of names meant to be acted on.
    """
    with pytest.raises(ValueError, match="negative"):
        negative([chunk(BERT, 0.9)]).expected_rank


def test_the_positives_and_negatives_split_the_list_without_losing_anyone() -> None:
    results = [
        result([chunk(ATTENTION, 0.6)]),
        negative([chunk(BERT, 0.9)]),
        result([], text="Q3"),
    ]

    assert len(positive_results(results)) == 2
    assert len(negative_results(results)) == 1
    assert len(positive_results(results)) + len(negative_results(results)) == len(results)


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


def test_summarize_top_scores_returns_none_when_there_are_no_scores() -> None:
    """None is one of the answers this has to be able to give.

    Not a dict of zeroes, which would read as a measured minimum of 0.0, and not an
    exception: an empty negative distribution is the best outcome a run can have, and
    raising on it would report a clean run as a broken one.
    """
    assert summarize_top_scores([result([])]) is None


def test_summary_keeps_four_decimals() -> None:
    """Two decimals is not enough here.

    The number a similarity threshold has to be set below is the minimum, and the real
    distribution's minimum is 0.3591 -- which rounding to 0.36 would blur into the same
    value as several neighbours.
    """
    results = [result([chunk(ATTENTION, 0.3591)]), result([chunk(ATTENTION, 0.6683)], text="Q2")]

    summary = summarize(results)

    assert summary["top_score_positives"]["min"] == 0.3591
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
        "positive_count",
        "negative_count",
        "retrieval_hit_rate",
        "retrieval_rank1_hit_rate",
        "avg_top_score",
        "top_score_positives",
        "top_score_negatives",
        "negative_scores_missing",
        "false_positive_rate_no_threshold",
        "rank1_misses",
        "negatives_above_positive_min",
        "threshold_separates",
        "separating_gap",
        "threshold_sweep",
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
    someone to fix three questions while four are broken.

    A negative is in the input on purpose. Its expected_rank is undefined rather than
    non-1, so a version that failed to filter would either raise here or -- comparing the
    null against 1 and finding them different -- list a question that was never expected to
    succeed as one that failed. Neither shows up while every input is a positive, which is
    all this test used to be.
    """
    results = [
        result([chunk(ATTENTION, 0.6)]),
        result([chunk(BERT, 0.5)], text="Q2"),
        result([chunk(BERT, 0.4), chunk(ATTENTION, 0.3)], text="Q3"),
        result([], text="Q4"),
        negative([chunk(BERT, 0.9)]),
    ]

    misses = len(describe_rank1_misses(results))
    positives = positive_results(results)

    assert misses == 3
    assert compute_rank1_hit_rate(results) == (len(positives) - misses) / len(positives)


# --- the rates, with negatives in the list ------------------------------------


def test_hit_rate_is_a_fraction_of_the_positives_not_of_every_question() -> None:
    """Adding a negative must not move the rate.

    A negative cannot hit, so counting it in the denominator would lower the rate by
    exactly its share -- a number that changes whenever the negative set is edited and says
    nothing about retrieval.
    """
    positives = [result([chunk(ATTENTION, 0.6)]), result([chunk(BERT, 0.5)], text="Q2")]

    assert compute_hit_rate(positives) == 0.5
    assert compute_hit_rate([*positives, negative([chunk(BERT, 0.9)])]) == 0.5


def test_rank1_hit_rate_is_a_fraction_of_the_positives_too() -> None:
    positives = [result([chunk(ATTENTION, 0.6)]), result([chunk(BERT, 0.5)], text="Q2")]

    assert compute_rank1_hit_rate(positives) == 0.5
    assert compute_rank1_hit_rate([*positives, negative([chunk(BERT, 0.9)])]) == 0.5


def test_the_rates_refuse_a_list_of_only_negatives() -> None:
    """Distinct from the empty-list guard beside it. An empty list is a caller that passed
    nothing; a list of negatives is a caller that passed the wrong thing."""
    with pytest.raises(ValueError, match="no positive questions"):
        compute_hit_rate([negative([chunk(BERT, 0.9)])])

    with pytest.raises(ValueError, match="no positive questions"):
        compute_rank1_hit_rate([negative([chunk(BERT, 0.9)])])


def test_avg_top_score_covers_the_positives_only() -> None:
    """Over both kinds it would fall as negatives are added, with retrieval unchanged.

    The fall would look like a retrieval regression and would really be an artefact of the
    evaluation set growing.
    """
    positives = [result([chunk(ATTENTION, 0.6)])]

    assert summarize(positives)["avg_top_score"] == 0.6
    assert summarize([*positives, negative([chunk(BERT, 0.1)])])["avg_top_score"] == 0.6


def test_describe_rank1_misses_ignores_negatives() -> None:
    """They cannot rank first or fail to, so they belong in neither the list nor its
    length."""
    results = [result([chunk(BERT, 0.6), chunk(ATTENTION, 0.5)]), negative([chunk(BERT, 0.9)])]

    assert [row["question"] for row in describe_rank1_misses(results)] == [
        "What is self-attention?"
    ]


# --- the negatives, and the threshold that is not there yet --------------------


def test_false_positive_rate_without_a_threshold_counts_every_negative_that_scored() -> None:
    """1.0 today, and reached by counting rather than asserted.

    Retrieval applies no threshold, so a negative that returned anything at all is a false
    positive. That is a real measurement of a real behaviour, and it will move by itself on
    the day retrieval starts declining -- which a hardcoded 1.0 would not.
    """
    results = [
        result([chunk(ATTENTION, 0.6)]),
        negative([chunk(BERT, 0.9)]),
        negative([], text="Q3"),
    ]

    summary = summarize(results)

    assert summary["false_positive_rate_no_threshold"] == 0.5
    assert summary["negative_scores_missing"] == 1


def test_false_positive_rate_is_none_when_there_are_no_negatives() -> None:
    """0.0 would read as "no negative was ever a false positive", which is a claim about
    negatives that do not exist. None says nothing was tested, which is the truth."""
    assert summarize([result([chunk(ATTENTION, 0.6)])])["false_positive_rate_no_threshold"] is None


def test_threshold_separates_the_two_ranges_when_they_do_not_overlap() -> None:
    assert threshold_separates([0.50, 0.60], [0.30, 0.40]) is True


def test_threshold_separates_is_false_when_the_ranges_overlap() -> None:
    assert threshold_separates([0.50, 0.60], [0.30, 0.55]) is False


@pytest.mark.parametrize(
    ("positives", "negatives"),
    [([0.50], []), ([], [0.30]), ([], [])],
)
def test_threshold_separates_is_none_when_a_side_is_missing(
    positives: list[float], negatives: list[float]
) -> None:
    """The tri-state, and the whole reason this is not a plain bool.

    False would report that a comparison failed when no comparison could be made. The two
    send a reader to different places -- False to the score distributions, None to the
    evaluation set -- and a summary read without negatives would otherwise look like a
    corpus that fails to separate.
    """
    assert threshold_separates(positives, negatives) is None


def test_separating_gap_is_the_distance_between_the_closest_pair() -> None:
    assert separating_gap([0.50, 0.60], [0.30, 0.40]) == pytest.approx(0.10)


def test_separating_gap_goes_negative_when_the_ranges_overlap() -> None:
    """The sign is the finding, so it is kept rather than clamped at zero."""
    assert separating_gap([0.50], [0.55]) == pytest.approx(-0.05)


def test_separating_gap_is_none_when_a_side_is_missing() -> None:
    assert separating_gap([0.50], []) is None
    assert separating_gap([], [0.30]) is None


def test_threshold_sweep_uses_the_observed_scores_as_its_thresholds() -> None:
    """Every row is then a decision the data distinguishes, and no grid spacing has to be
    chosen or defended."""
    sweep = threshold_sweep([0.5, 0.7], [0.2, 0.7])

    assert [row["threshold"] for row in sweep] == [0.2, 0.5, 0.7]


def test_threshold_sweep_walks_from_everything_kept_to_everything_rejected() -> None:
    """Both endpoints present, so a row exists for any operating point worth discussing."""
    sweep = threshold_sweep([0.5, 0.7], [0.2, 0.4])

    assert sweep[0] == {"threshold": 0.2, "positives_kept": 2, "negatives_rejected": 0}
    assert sweep[-1] == {"threshold": 0.7, "positives_kept": 1, "negatives_rejected": 2}


def test_threshold_sweep_decides_a_tie_the_same_way_on_both_sides() -> None:
    """A score equal to the threshold is kept on both sides, never kept on one and
    rejected on the other.

    Equality is certain here rather than a rare edge, because the thresholds are drawn from
    the scores. Writing the negative side as ``<=`` while the positive side is ``>=`` would
    produce a row that keeps a negative and reports it rejected on the same line -- and a
    reader taking the positive column at face value would conclude the threshold works.
    """
    (row,) = threshold_sweep([0.5], [0.5])

    assert row == {"threshold": 0.5, "positives_kept": 1, "negatives_rejected": 0}


def test_threshold_sweep_reports_its_thresholds_unrounded() -> None:
    """A threshold is a score, and rounding it can put it above itself.

    0.42006 rounds to 0.4201, which is greater -- so a reader comparing a row against the
    scores it was built from would find a score below the threshold that the row counted as
    kept. The rest of the summary rounds freely for the same reason it rounds here at all;
    this column is the one place where rounding changes what the number means.
    """
    sweep = threshold_sweep([0.42006], [0.1])

    assert max(row["threshold"] for row in sweep) == 0.42006


def test_threshold_sweep_is_none_when_a_side_is_missing() -> None:
    """A trade-off needs two sides. An empty list would read as a sweep that ran and found
    nothing to say."""
    assert threshold_sweep([0.5], []) is None
    assert threshold_sweep([], [0.5]) is None


def test_describe_negatives_above_the_floor_names_the_ones_that_outscored_it() -> None:
    """The actionable part of a failed separation.

    Each of these is a question the corpus cannot answer that retrieval nonetheless scored
    as confidently as the least confident question it can. Naming them separates two causes
    a single number merges -- scores too close because the corpus nearly answers these,
    versus an embedding space that does not separate them -- and only reading the questions
    can say which it is.
    """
    results = [
        result([chunk(ATTENTION, 0.40)]),
        negative([chunk(BERT, 0.55)], text="outscored it"),
        negative([chunk(BERT, 0.10)], text="comfortably below"),
    ]

    above = describe_negatives_above(
        negative_results(results), collect_top_scores(positive_results(results))
    )

    assert [row["question"] for row in above] == ["outscored it"]


def test_describe_negatives_above_includes_one_sitting_exactly_on_the_floor() -> None:
    """At the floor, not merely above it.

    A threshold placed at the lowest positive score keeps a negative that ties it, so such
    a negative belongs in the list of things that threshold fails to turn away.
    """
    above = describe_negatives_above([negative([chunk(BERT, 0.40)])], [0.40])

    assert len(above) == 1


def test_describe_negatives_above_is_empty_without_positives() -> None:
    """No floor to compare against, so nothing is above it.

    An empty list rather than every negative, which is what comparing against an absent
    floor would produce.
    """
    assert describe_negatives_above([negative([chunk(BERT, 0.9)])], []) == []


def test_summarize_refuses_a_run_with_no_positives_before_it_computes_anything() -> None:
    """The guard runs first, so the error names the evaluation set and not the statistics
    module.

    summarize builds a dict literal, and a literal evaluates its values in source order --
    so the same guard written as one of those values would run only after avg_top_score had
    been computed, well before it in the literal. On a list of negatives that raises
    ``StatisticsError: mean requires at least one data point``, which main() catches as a
    ValueError and prints, sending the reader to look for a bug in a mean.
    """
    with pytest.raises(ValueError, match="no positive questions"):
        summarize([negative([chunk(BERT, 0.9)])])


# --- the real evaluation set --------------------------------------------------


@pytest.fixture(scope="module")
def real_questions() -> list[EvalQuestion]:
    return load_eval_questions(QUESTIONS_PATH)


def test_the_shipped_evaluation_set_loads(real_questions) -> None:
    assert len(real_questions) >= 20


def test_the_shipped_evaluation_set_has_both_kinds_of_question(real_questions) -> None:
    """Both kinds, because either alone silently disables half the summary.

    With no negatives the sweep has nothing to price and ``threshold_separates`` is null;
    with no positives every rate is undefined and the file does not load at all. Neither
    absence announces itself in the output -- a summary with no negatives looks exactly
    like a corpus that answers everything -- so the file is checked here instead.
    """
    assert positive_questions(real_questions) != []
    assert [q for q in real_questions if q.is_negative] != []


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

    # Positives only. A negative's expected_source is None, which is not an unknown title
    # but the absence of one -- and mixing it into this set would also make the sort below
    # compare None against str.
    unknown = sorted({q.expected_source for q in real_questions if not q.is_negative} - titles)

    assert unknown == []


def test_every_expected_term_appears_in_the_document_it_is_labelled_with(real_questions) -> None:
    """Same guard, for the terms.

    They are not scored yet, so a term that no longer occurs costs nothing today -- and
    would cost a silent re-reading of every paper on the day an answer-quality metric
    starts using them.
    """
    files = {paper.title: CORPUS_DIR / f"{paper.slug}.txt" for paper in PAPERS}

    missing = []
    for question in positive_questions(real_questions):
        text = files[question.expected_source].read_text(encoding="utf-8")
        missing += [
            (question.expected_source, term)
            for term in question.required_terms
            if term not in text
        ]

    assert missing == []


def test_no_forbidden_term_occurs_anywhere_in_the_corpus(real_questions) -> None:
    """The negative half of the corpus check, and the only mechanical check there is.

    A question may be written claiming the corpus cannot answer it, but that claim is about
    files sitting in another directory, and the line loads and scores whether or not it is
    true. It cannot be tested by rereading the question: an answer is allowed to be phrased
    any way at all, and ten questions against twenty papers is more than a careful person
    reliably holds. The term can be tested, so it is.

    Necessary and not sufficient, and the gap is worth stating plainly rather than leaving
    implied. A term absent from every file proves that term is absent; it does not prove
    the question is unanswerable, because the corpus may answer it in words the term does
    not cover. This guard can fail a negative that is definitely wrong. It cannot certify
    one as definitely right, and nothing in this suite can.
    """
    files = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(CORPUS_DIR.glob("*.txt"))
    }

    found = [
        (question.question, term, name)
        for question in real_questions
        for term in question.forbidden_terms
        for name, text in files.items()
        if term in text
    ]

    assert found == []


def test_the_evaluation_set_labels_the_corpus_the_manifest_describes(real_questions) -> None:
    """sources.json and the .txt files on disk must describe the same twenty papers.

    The manifest is committed; the papers it names are fetched into place by
    scripts/fetch_corpus.py, so this is also what fails when someone has the evaluation set
    but has not run that yet.

    Read rather than assumed: the titles in this module and in the loader both come from
    it, so a manifest listing a file that is not there would make the guard above pass
    against a corpus that no longer exists.
    """
    manifest = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))

    assert {entry["title"] for entry in manifest["papers"]} == {paper.title for paper in PAPERS}
    assert all((CORPUS_DIR / entry["file"]).exists() for entry in manifest["papers"])
