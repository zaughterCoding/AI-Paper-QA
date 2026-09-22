"""Run the evaluation questions through retrieval and report the metrics.

    python scripts\\evaluate.py [--top-k 5] [--questions eval/questions.jsonl]

Prints one JSON object to stdout and nothing else, so the run can be redirected to a file
and diffed against the previous one. Progress and failures go to stderr.

Two metrics come from the plan:

    retrieval_hit_rate  fraction of questions whose expected source is among the returned
                        sources, at any rank.
    avg_top_score       mean score of the top-ranked chunk.

Two more are reported beside them, for reasons found while building the evaluation set:

    retrieval_rank1_hit_rate  the same fraction, but counting only questions whose expected
                              source is ranked first. The loose metric is not a weaker
                              version of this one -- on this corpus it reads 1.0 at
                              ``top_k=20`` while nine questions have the wrong document at
                              rank 1, so a saturated 1.0 says nothing about ranking quality.
    top_score_positives       min / median / max beside the mean. A similarity threshold
                              has to be set below the lowest score an answerable question
                              produces, and a mean averages that edge away: the 25 positives
                              here span 0.4029 to 0.6852 around a mean of 0.5754.

Read ``retrieval_hit_rate`` with its random baseline in mind, because the baseline is not
zero. The corpus holds 731 chunks across 20 documents, so picking ``top_k`` chunks at
random finds the right document about 0.188 of the time at ``top_k=5`` and about 0.568 of
the time at ``top_k=20``. A rate near 0.2 is therefore not "failing"; it is what chance
produces, and a rate of 1.0 at a wide ``top_k`` may say nothing at all. The baseline falls
as documents are added and rises with ``top_k``, so the rate is comparable only between
runs over the same corpus with the same ``top_k``.

``retrieval_rank1_hit_rate`` has no such dependence: rank 1 is the same chunk whatever
``top_k`` is, so raising the budget cannot move it. That is what makes it the useful one --
on this corpus it stays at 0.64 for ``top_k`` of 1, 5 and 20, so the nine questions it
marks as failures are not short of budget and raising ``top_k`` will not fix them.

Three of those nine need reading with care, because the questions were written against a
five-document corpus and the corpus now holds twenty near-neighbours. For three of them the
document that displaced the labelled one states the same fact in its own words: ELECTRA
gives the 15% masking and the 10%/10% replacement, RoBERTa defines next sentence prediction,
and RocketQA states that the first token is the encoders' output representation. Those
questions have two defensible answers and this rate counts retrieving either as a failure.
They are left unchanged rather than edited or deleted -- no term can separate two documents
that state the same fact, and rewriting a question because its rank moved would make the
metric measure the questions instead of the retriever.

Questions are of two kinds. One carrying an ``expected_source`` is a *positive*: the corpus
contains an answer, and the metrics above ask whether retrieval found it. One whose
``expected_source`` is ``null`` is a *negative*: the corpus does not contain an answer, so
nothing can be found, and the only thing to measure is the score retrieval reported anyway.

Negatives were added because the one decision this script exists to inform could not be made
without them. A similarity threshold has to sit low enough that an answerable question still
clears it -- which the positive scores alone can say -- and high enough that an unanswerable
one does not, which they cannot. With one side measured, any threshold is a guess about the
other, and the guess is never checked.

The threshold columns describe scores, not a decision, and they describe a behaviour that
does not exist yet. Retrieval applies no threshold by design (see the note in
``app/services/retrieval.py``), and the answer layer only declines when retrieval returns
nothing at all, which cannot happen while the corpus is non-empty. ``threshold_sweep`` is
therefore a what-if: read a row as "if retrieval declined below this score, this many
answerable questions would be lost and this many unanswerable ones turned away". Three
things it does not model:

    * it gates whole questions on their top score, not individual chunks. Filtering chunks
      would change which of ranks 2..k survive and so change ``retrieval_hit_rate``, about
      which the sweep says nothing;
    * it is invariant to ``top_k``, because the top score is. Re-running it at a wider
      ``top_k`` to get a better curve would produce the same curve;
    * the positive side is a rank statistic of however many questions the file holds. The
      lowest positive score moves down the moment a harder question is added, and a
      threshold set exactly on it is set on the most optimistic value the sample supports.

No key here names a recommended threshold, and none should be added. A threshold in the
output would stop being evidence and start being a default.

The questions' ``expected_terms`` field is loaded and validated but not scored. Checking
whether an answer contains those terms needs the answer, which means calling the LLM, and
neither metric here does. The field is kept rather than dropped so the loader does not
silently discard an annotation that a later answer-quality metric will want. Its meaning
inverts for negatives, where it lists terms that would prove the corpus *does* hold the
answer -- which is what makes a negative example checkable against the corpus at all.
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.database import SessionLocal
from app.rag.embeddings import get_embedding_client
from app.repositories.chunks import RetrievedChunk
from app.services.retrieval import DEFAULT_TOP_K, MAX_TOP_K, RetrievalService

# Derived from __file__ so the default is the same from any cwd.
QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"

# Every field the loader understands. A line carrying anything else is rejected rather
# than read for the fields it does have: a mistyped key would otherwise leave a label the
# author believed was in use silently unscored, which is how ``expected_terms`` went
# unused through the whole task that introduced it.
REQUIRED_FIELDS = ("question", "expected_terms", "expected_source")

# Scores are rounded for readability, not to two decimals: the interesting figure is the
# lowest score an answerable question produces, and rounding 0.3591 to 0.36 loses the part
# that a threshold decision turns on. Thresholds in ``threshold_sweep`` are deliberately
# left unrounded -- a threshold is a score, and rounding it would let a row report its own
# score as falling below itself.
SCORE_PRECISION = 4


@dataclass(frozen=True)
class EvalQuestion:
    """One benchmark question and the labels attached to it.

    ``expected_source`` is a document title -- the same string as ``documents.title``, not
    a file name or a slug -- because that is what a retrieval result carries and therefore
    what can be compared without a join. It is ``None`` for a negative question, which is
    the whole of what marks one: the field that says which document should be found says
    instead that none should.

    Marking it implicitly, by ``None``, has a cost worth naming. A ``null`` typed by mistake
    where a title was meant inverts the question: it reads as "no document should be found"
    and turns the terms from things that must occur into things that must not. The file
    still loads, the metrics still compute, and nothing in this module can tell. What does
    catch it is the corpus check in the test suite, which reads every labelled document and
    every term; an explicit ``is_negative`` field would remove the ambiguity but would also
    make ``expected_source: null`` with ``is_negative: false`` representable, which is a
    worse state than the one it fixes.
    """

    question: str
    expected_terms: list[str]
    expected_source: str | None

    @property
    def is_negative(self) -> bool:
        """True when the corpus is not expected to answer this question."""
        return self.expected_source is None

    @property
    def required_terms(self) -> list[str]:
        """Terms that must occur in ``expected_source``. Empty for a negative.

        Defined as a pair with ``forbidden_terms`` so that a caller cannot apply one
        question's terms on the wrong side: there is no single ``expected_terms`` to reach
        for that means both things.
        """
        return [] if self.is_negative else self.expected_terms

    @property
    def forbidden_terms(self) -> list[str]:
        """Terms whose presence would prove the corpus answers this question.

        Empty for a positive. A negative's terms are not an answer key -- they are a
        falsifier. "What learning rate does the paper recommend?" cannot be checked against
        a corpus, because the answer may be phrased any number of ways; "does the corpus
        contain the token ``warmup_steps`` anywhere" can be, and a hit there means the
        negative is not one.
        """
        return self.expected_terms if self.is_negative else []


@dataclass(frozen=True)
class EvalResult:
    """One question and the chunks retrieval returned for it.

    The chunks are kept whole rather than reduced to a list of titles, so rank survives:
    list order is rank order, and position in ``retrieved`` is the only place rank is
    recorded. Every metric below reads it from there.
    """

    question: EvalQuestion
    retrieved: list[RetrievedChunk]

    @property
    def expected_rank(self) -> int | None:
        """1-based rank of the expected source, or None when it was not returned.

        The two outcomes are kept apart instead of collapsed into a rank of 0 or a large
        number: "returned, but ranked third" and "not returned at all" are different
        failures with different fixes, and the counters that pick them apart are already
        there.

        Undefined, and so an error, for a negative question. A negative has no expected
        source, and the comparison below would quietly answer the wrong question: a chunk
        title is never ``None``, so the loop would fall through and return ``None`` --
        which every caller here reads as "the expected source was not found", i.e. as a
        miss. Computing it would be a silent wrong answer rather than a missing one, and
        the rate built on it would count each negative as a retrieval failure.
        """
        if self.question.is_negative:
            raise ValueError(
                f"expected_rank is undefined for the negative question "
                f"{self.question.question!r}: it has no expected source to rank. "
                "Separate positives from negatives before asking for a rank."
            )

        for position, chunk in enumerate(self.retrieved, start=1):
            if chunk.title == self.question.expected_source:
                return position
        return None

    @property
    def top_score(self) -> float | None:
        """Score of the best chunk, or None when nothing came back.

        None rather than 0.0: no chunk means no score, whereas 0.0 would be a measurement
        -- one that would drag the mean and the minimum down to a value no real result can
        produce, and quietly flatten the distribution a threshold would be read off.
        """
        return self.retrieved[0].score if self.retrieved else None


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    """Read a JSONL evaluation file, one question per line.

    Blank lines are skipped so a trailing newline is not an error. Every other problem
    raises ``ValueError`` naming the line number: the file is written and edited by hand,
    and a loader that skips a line it cannot parse turns a typo into a quietly smaller
    benchmark.

    Duplicate questions are rejected. They would be scored twice, which weights them twice
    in every rate here -- a distortion nothing downstream can see.

    A file of nothing but negatives is rejected too, and for a related reason. Every rate
    here divides by the number of positives, and the summary refuses to run without one, so
    such a file could only ever produce an error further from its cause.
    """
    path = Path(path)

    # splitlines rather than split("\n"): the file is committed with LF endings and checked
    # out with CRLF on Windows, and splitlines handles both without leaving a stray \r
    # inside the last field of each line.
    lines = path.read_text(encoding="utf-8").splitlines()

    questions: list[EvalQuestion] = []
    seen: set[str] = set()

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: not valid JSON: {error}") from error

        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object, got {type(row).__name__}")

        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}:{number}: missing field(s): {', '.join(missing)}")

        unexpected = sorted(set(row) - set(REQUIRED_FIELDS))
        if unexpected:
            raise ValueError(
                f"{path}:{number}: unknown field(s): {', '.join(unexpected)}. "
                f"Known fields are {', '.join(REQUIRED_FIELDS)}; an unrecognized one would "
                "never be read, so it is rejected instead of ignored. A question the corpus "
                'cannot answer needs no new field -- write "expected_source": null.'
            )

        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{number}: question must be a non-empty string")

        if question in seen:
            raise ValueError(f"{path}:{number}: duplicate question: {question!r}")
        seen.add(question)

        terms = row["expected_terms"]
        # The isinstance check is not decoration: a bare string iterates into characters,
        # so "self-attention" would be stored as thirteen single-character terms and every
        # later use of them would be quietly wrong.
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"{path}:{number}: expected_terms must be a non-empty list of strings")
        if not all(isinstance(term, str) and term.strip() for term in terms):
            raise ValueError(f"{path}:{number}: every expected_terms entry must be a non-empty string")

        source = row["expected_source"]
        # None is the negative marker and is the only value let through that is not a
        # title. The empty string is neither: it is not a way of saying "no document",
        # because that is what null is for, and it is not a title either -- it would simply
        # never equal any document's, so the question would score as a permanent
        # unexplained miss.
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError(
                f"{path}:{number}: expected_source must be null (the corpus does not answer "
                f"this question) or a non-empty string naming a document title, got {source!r}"
            )

        questions.append(
            EvalQuestion(question=question, expected_terms=terms, expected_source=source)
        )

    if not questions:
        # An empty file is not a benchmark that scores nothing; it is a broken benchmark.
        # Returning [] would push that to the metrics, where it reads as a division guard
        # rather than as the misconfiguration it is.
        raise ValueError(f"{path}: no questions found")

    positives = sum(1 for question in questions if not question.is_negative)
    if not positives:
        raise ValueError(
            f"{path}: no positive questions found; all {len(questions)} have a null "
            "expected_source. The rates here are fractions of the positives, so a file "
            "with none of them has nothing to score."
        )

    _reject_terms_used_on_both_sides(path, questions)

    return questions


def _reject_terms_used_on_both_sides(path: Path, questions: list[EvalQuestion]) -> None:
    """Refuse a term that one question requires and another forbids.

    The two uses of ``expected_terms`` contradict each other, and the corpus check that
    reads them cannot satisfy both: whichever document is inspected, one of the two
    questions is wrong. Failing here names both of them, which is far more use than a test
    reporting that some term is "missing" from one file and "present" in another.

    Exact string comparison, matching how the terms are checked against the corpus.
    """
    required_by: dict[str, str] = {}
    for question in questions:
        for term in question.required_terms:
            required_by.setdefault(term, question.question)

    for question in questions:
        for term in question.forbidden_terms:
            if term in required_by:
                raise ValueError(
                    f"{path}: term {term!r} is required by the question "
                    f"{required_by[term]!r} and forbidden by {question.question!r}. "
                    "One of the two is wrong: the term cannot both occur in the corpus "
                    "and not occur in it."
                )


def positive_results(results: list[EvalResult]) -> list[EvalResult]:
    """The results for questions the corpus is expected to answer."""
    return [result for result in results if not result.question.is_negative]


def negative_results(results: list[EvalResult]) -> list[EvalResult]:
    """The results for questions the corpus is expected not to answer."""
    return [result for result in results if result.question.is_negative]


def compute_hit_rate(results: list[EvalResult]) -> float:
    """Fraction of questions whose expected source came back, at any rank.

    This is the plan's ``retrieval_hit_rate``. It is easy to pass: at ``top_k=20`` on this
    corpus every question retrieves its own source somewhere among the twenty, so the rate
    reads 1.0 while nine of them have the wrong document at rank 1. It cannot tell a
    well-ranked result from a poorly ranked one. ``compute_rank1_hit_rate`` is the version
    that still moves.

    Negatives are dropped inside rather than left to the caller. A negative cannot hit, so
    counting it would lower the rate by exactly its share -- a number that moves when the
    negative set changes and says nothing about retrieval. Filtering here rather than at
    the call site means a caller who forgets gets the right rate anyway, since a wrong
    denominator is not something this function can detect.
    """
    if not results:
        raise ValueError("no results to score")

    positives = positive_results(results)
    if not positives:
        # Distinct from the guard above: an empty list is a caller that passed nothing, a
        # list of negatives is a caller that passed the wrong thing. Same 0.0 either way if
        # it were allowed through, and neither is a measurement.
        raise ValueError("no positive questions to score; every result is a negative")

    hits = sum(1 for result in positives if result.expected_rank is not None)
    return hits / len(positives)


def compute_rank1_hit_rate(results: list[EvalResult]) -> float:
    """Fraction of questions whose expected source was ranked first.

    Also known as recall@1. This is the metric that separates runs the loose rate calls
    identical, and the one worth watching when ``top_k`` or the chunking changes.

    Negatives are dropped for the same reason as above, and it matters more here: a
    negative's ``expected_rank`` is an error rather than a number, so it could not be
    counted even if the denominator were meant to include it.
    """
    if not results:
        raise ValueError("no results to score")

    positives = positive_results(results)
    if not positives:
        raise ValueError("no positive questions to score; every result is a negative")

    hits = sum(1 for result in positives if result.expected_rank == 1)
    return hits / len(positives)


def collect_top_scores(results: list[EvalResult]) -> list[float]:
    """The top-chunk score of every question that returned something.

    Questions that returned nothing are left out rather than counted as 0.0, and the caller
    reports the length of this list beside the question counts so the omission is visible
    in the output instead of being folded into the statistics.
    """
    return [result.top_score for result in results if result.top_score is not None]


def summarize_top_scores(results: list[EvalResult]) -> dict | None:
    """Distribution of top-chunk scores, or None when there are no scores to describe.

    ``count`` is reported beside the statistics because it is the only sign that a
    question returning nothing was left out. Without it the statistics would describe fewer
    questions than the counts say, with nothing in the output to show it.

    None rather than an exception when the list is empty, because "no distribution" is one
    of the answers this has to be able to give. The negative distribution is empty exactly
    when every negative scored nothing -- the best possible outcome, and one that must not
    be reported as a crash or as a dict of zeroes, which would read as a measured minimum
    of 0.0.
    """
    scores = collect_top_scores(results)

    if not scores:
        return None

    return {
        "count": len(scores),
        "min": round(min(scores), SCORE_PRECISION),
        "median": round(statistics.median(scores), SCORE_PRECISION),
        "max": round(max(scores), SCORE_PRECISION),
    }


def threshold_separates(positive_scores: list[float], negative_scores: list[float]) -> bool | None:
    """Whether some threshold keeps every positive and rejects every negative.

    True when the lowest positive score beats the highest negative one, which is exactly
    when such a threshold exists: any value in the gap does it. False when the ranges
    overlap, so no threshold can be right for both sides at once.

    None when either side has no scores, and this is the point of the tri-state. A plain
    bool would answer False for a run with no negatives -- reporting that a comparison
    failed when no comparison could be made. False and None send a reader to different
    places: one to the score distributions, the other to the evaluation set.
    """
    if not positive_scores or not negative_scores:
        return None
    return min(positive_scores) > max(negative_scores)


def separating_gap(positive_scores: list[float], negative_scores: list[float]) -> float | None:
    """Lowest positive score minus highest negative score, or None if either side is empty.

    Negative means the ranges overlap. Left unrounded: it is read to decide whether a gap
    is wide enough to place a threshold in, and at four decimals a gap of 0.00003 and one
    of 0.00004 look equally usable while a gap of 0.00001 disappears entirely.
    """
    if not positive_scores or not negative_scores:
        return None
    return min(positive_scores) - max(negative_scores)


def threshold_sweep(
    positive_scores: list[float], negative_scores: list[float]
) -> list[dict] | None:
    """Every threshold the data produces, with what it would keep and reject.

    The candidate thresholds are the observed scores themselves, pooled and deduplicated,
    rather than an evenly spaced grid. Every row is then a decision the scores actually
    distinguish, and no arbitrary constant has to be chosen or defended.

    Both counts use one rule: a score is kept when it is greater than or equal to the
    threshold and rejected when it is below. Splitting the two sides across the boundary --
    positives at ``>`` and negatives at ``<=`` -- would let a row keep a negative and
    reject it in the same breath whenever a score equalled the threshold, and equality is
    guaranteed to occur here because the thresholds are drawn from the scores.

    Counts, not fractions: a fraction of a small set is hard to read back into a count, and
    rounding one twice loses more. The denominators are printed once, at the top level.

    None when either side has no scores, since a trade-off needs two sides. An empty list
    would read as a sweep that ran and found nothing to say.
    """
    if not positive_scores or not negative_scores:
        return None

    thresholds = sorted(set(positive_scores) | set(negative_scores))

    return [
        {
            "threshold": threshold,
            "positives_kept": sum(1 for score in positive_scores if score >= threshold),
            "negatives_rejected": sum(1 for score in negative_scores if score < threshold),
        }
        for threshold in thresholds
    ]


def describe_negatives_above(
    negatives: list[EvalResult], positive_scores: list[float]
) -> list[dict]:
    """The negative questions that scored at or above the lowest positive one.

    The actionable part of a failed separation: each of these is a question the corpus
    cannot answer that retrieval nonetheless scored as confidently as the least confident
    question it can. Naming them separates two causes that a single number merges -- the
    scores may be too close because the corpus really does nearly answer these, or because
    the embedding space does not separate them, and only reading the questions can tell
    which.

    Compared against the unrounded minimum. ``top_score_positives["min"]`` is rounded to
    four decimals and can differ in the last digit, so a question sitting exactly on the
    floor could appear here without appearing to reach it there.
    """
    if not positive_scores:
        return []

    floor = min(positive_scores)

    return [
        {
            "question": result.question.question,
            "top_score": (
                round(result.top_score, SCORE_PRECISION)
                if result.top_score is not None
                else None
            ),
            "ranked_first": result.retrieved[0].title if result.retrieved else None,
        }
        for result in negatives
        if result.top_score is not None and result.top_score >= floor
    ]


def summarize(results: list[EvalResult]) -> dict:
    """The JSON summary: the plan's two metrics plus the ones that qualify them."""
    # Both guards come first, before the dict below, and that ordering is load-bearing.
    # A dict literal evaluates its values in source order, so a guard written as one of
    # them would run only after the values before it had already been computed -- and
    # ``avg_top_score`` is early in the literal. On a run with no positives it would raise
    # ``StatisticsError: mean requires at least one data point``, caught by main() as a
    # ValueError, and the reader would be sent to the statistics module for what is really
    # a fact about the evaluation set.
    if not results:
        raise ValueError("no results to score")

    positives = positive_results(results)
    if not positives:
        raise ValueError("no positive questions to score; every result is a negative")

    negatives = negative_results(results)
    positive_scores = collect_top_scores(positives)
    negative_scores = collect_top_scores(negatives)

    return {
        "question_count": len(results),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        # Both rates are fractions of the positives, not of question_count. The negatives
        # are in the file to be measured against the positives, not to dilute them.
        "retrieval_hit_rate": round(compute_hit_rate(results), SCORE_PRECISION),
        "retrieval_rank1_hit_rate": round(compute_rank1_hit_rate(results), SCORE_PRECISION),
        # Positives only. Over both kinds this would fall as negatives are added, with
        # retrieval unchanged, and the drop would mean nothing.
        "avg_top_score": (
            round(statistics.mean(positive_scores), SCORE_PRECISION) if positive_scores else None
        ),
        "top_score_positives": summarize_top_scores(positives),
        "top_score_negatives": summarize_top_scores(negatives),
        # The negatives that scored and the ones that did not are counted separately, so
        # the distribution above can be read knowing what is missing from it.
        "negative_scores_missing": len(negatives) - len(negative_scores),
        # The only false-positive rate that involves no choice: retrieval applies no
        # threshold, so every negative that returned anything is a false positive. It is
        # 1.0 today and will stay 1.0 until retrieval starts declining, at which point the
        # number moves on its own. A rate under a chosen threshold is a property of the
        # choice, not of the system, and is left to threshold_sweep to show.
        "false_positive_rate_no_threshold": (
            round(len(negative_scores) / len(negatives), SCORE_PRECISION) if negatives else None
        ),
        "rank1_misses": describe_rank1_misses(results),
        "negatives_above_positive_min": describe_negatives_above(negatives, positive_scores),
        "threshold_separates": threshold_separates(positive_scores, negative_scores),
        "separating_gap": separating_gap(positive_scores, negative_scores),
        "threshold_sweep": threshold_sweep(positive_scores, negative_scores),
    }


def describe_rank1_misses(results: list[EvalResult]) -> list[dict]:
    """Name the questions whose expected source was not ranked first.

    A rate on its own says how many failed but not which, and "which" is the only part
    that can be acted on: the nine that fail here point at concrete retrieval problems,
    while the rate they are counted in points at nothing by itself -- at ``top_k=20``
    ``retrieval_hit_rate`` reads 1.0 with all nine still failing. ``ranked_first`` is the
    document that beat the expected one, and ``expected_rank`` is null when it was not
    returned at all.

    Questions that returned nothing are listed too, with both other fields null: they
    missed rank 1 by the widest margin there is, and leaving them out would make this list
    shorter than the failures the rates count.

    Negatives are dropped, and not only because they cannot rank. ``expected_rank`` raises
    for one, so a negative reaching this comprehension would end the run; and if it did
    not, a null rank differs from 1 and every negative would be listed as a failure of a
    question that was never expected to succeed.
    """
    return [
        {
            "question": result.question.question,
            "expected_source": result.question.expected_source,
            "expected_rank": result.expected_rank,
            "ranked_first": result.retrieved[0].title if result.retrieved else None,
        }
        for result in positive_results(results)
        if result.expected_rank != 1
    ]


def evaluate(
    service: RetrievalService, questions: list[EvalQuestion], top_k: int
) -> list[EvalResult]:
    """Retrieve for every question, in order, keeping the results in the same order.

    Retrieval goes through ``RetrievalService`` rather than the repository so the run
    measures the component that actually serves requests, validation included.

    Negative questions go through the same call as positives. They are not a separate mode
    of retrieval -- the point of them is that nothing about the request marks them as
    unanswerable, so the score that comes back is the one a real user would get.
    """
    return [
        EvalResult(question=question, retrieved=service.retrieve(question.question, top_k))
        for question in questions
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation set through retrieval")
    parser.add_argument(
        "--questions",
        default=str(QUESTIONS_PATH),
        help="JSONL file of evaluation questions (default: eval/questions.jsonl)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"chunks to retrieve per question (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    # Checked before the model loads: encoding a few dozen questions is cheap, loading
    # several hundred MB first and then raising is not.
    if not 1 <= args.top_k <= MAX_TOP_K:
        print(f"--top-k must be between 1 and {MAX_TOP_K}, got {args.top_k}", file=sys.stderr)
        return 1

    try:
        questions = load_eval_questions(args.questions)
    except (OSError, ValueError) as error:
        print(f"Could not read the evaluation set: {error}", file=sys.stderr)
        return 1

    session = SessionLocal()
    try:
        service = RetrievalService(session, get_embedding_client())
        results = evaluate(service, questions, args.top_k)
    finally:
        session.close()

    if not any(result.retrieved for result in results):
        # Not a score of zero: nothing was retrieved for any question, which means the
        # corpus has no vectors. Reporting 0.0 here would look like a retrieval result and
        # send the reader into the retrieval code instead of to scripts\\index_pending.py.
        print(
            "No chunks were retrieved for any question -- the corpus is empty or has no "
            "embeddings. Import the corpus, then run scripts\\index_pending.py.",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summarize(results), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
