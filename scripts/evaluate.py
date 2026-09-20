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
                              version of this one -- on this corpus it reads 1.0 while
                              three questions have the wrong document at rank 1, so a
                              saturated 1.0 says nothing about ranking quality.
    top_score                 min / median / max beside the mean. A similarity threshold
                              has to be set below the lowest score an answerable question
                              produces, and a mean averages that edge away: 25 questions
                              here span 0.3591 to 0.6678 around a mean of 0.5568.

Read ``retrieval_hit_rate`` with its random baseline in mind, because the baseline is not
zero. The corpus holds 149 chunks across 5 documents, so picking ``top_k`` chunks at random
finds the right document about 0.675 of the time at ``top_k=5`` and about 0.992 of the time
at ``top_k=20``. A rate near 0.7 is therefore not "almost working"; it is what chance
produces, and a rate of 1.0 at a wide ``top_k`` may say nothing at all. The baseline falls
as documents are added and rises with ``top_k``, so the rate is comparable only between
runs over the same corpus with the same ``top_k``.

``retrieval_rank1_hit_rate`` has no such dependence: rank 1 is the same chunk whatever
``top_k`` is, so raising the budget cannot move it. That is what makes it the useful one --
on this corpus it stays at 0.88 for ``top_k`` of 1, 5 and 20, so the three questions it
marks as failures are not short of budget and raising ``top_k`` will not fix them.

The questions' ``expected_terms`` field is loaded and validated but not scored. Checking
whether an answer contains those terms needs the answer, which means calling the LLM, and
neither metric here does. The field is kept rather than dropped so the loader does not
silently discard an annotation that a later answer-quality metric will want.
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
# that a threshold decision turns on.
SCORE_PRECISION = 4


@dataclass(frozen=True)
class EvalQuestion:
    """One benchmark question and the labels attached to it.

    ``expected_source`` is a document title -- the same string as ``documents.title``, not
    a file name or a slug -- because that is what a retrieval result carries and therefore
    what can be compared without a join.
    """

    question: str
    expected_terms: list[str]
    expected_source: str


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
        """
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
                "never be read, so it is rejected instead of ignored."
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
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{path}:{number}: expected_source must be a non-empty string")

        questions.append(
            EvalQuestion(question=question, expected_terms=terms, expected_source=source)
        )

    if not questions:
        # An empty file is not a benchmark that scores nothing; it is a broken benchmark.
        # Returning [] would push that to the metrics, where it reads as a division guard
        # rather than as the misconfiguration it is.
        raise ValueError(f"{path}: no questions found")

    return questions


def compute_hit_rate(results: list[EvalResult]) -> float:
    """Fraction of questions whose expected source came back, at any rank.

    This is the plan's ``retrieval_hit_rate``. It is easy to pass: with five documents and
    ``top_k=5`` almost every question retrieves its own source somewhere in the five, so
    the rate sits at or near 1.0 and cannot tell a well-ranked result from a poorly ranked
    one. ``compute_rank1_hit_rate`` is the version that still moves.
    """
    if not results:
        raise ValueError("no results to score")

    hits = sum(1 for result in results if result.expected_rank is not None)
    return hits / len(results)


def compute_rank1_hit_rate(results: list[EvalResult]) -> float:
    """Fraction of questions whose expected source was ranked first.

    Also known as recall@1. This is the metric that separates runs the loose rate calls
    identical, and the one worth watching when ``top_k`` or the chunking changes.
    """
    if not results:
        raise ValueError("no results to score")

    hits = sum(1 for result in results if result.expected_rank == 1)
    return hits / len(results)


def collect_top_scores(results: list[EvalResult]) -> list[float]:
    """The top-chunk score of every question that returned something.

    Questions that returned nothing are left out rather than counted as 0.0, and the
    caller reports the length of this list beside ``question_count`` so the omission is
    visible in the output instead of being folded into the statistics.
    """
    return [result.top_score for result in results if result.top_score is not None]


def summarize_top_scores(results: list[EvalResult]) -> dict:
    """Distribution of top-chunk scores, plus how many questions are in it.

    ``count`` is reported beside the statistics because it is the only sign that a
    question returning nothing was left out. Without it the statistics would describe
    fewer questions than ``question_count`` with nothing in the output to say so.
    """
    scores = collect_top_scores(results)

    if not scores:
        # main() has already refused a run where nothing was retrieved; this keeps the
        # function honest when called on its own.
        raise ValueError("no scores to summarize")

    return {
        "count": len(scores),
        "min": round(min(scores), SCORE_PRECISION),
        "median": round(statistics.median(scores), SCORE_PRECISION),
        "max": round(max(scores), SCORE_PRECISION),
    }


def summarize(results: list[EvalResult]) -> dict:
    """The JSON summary: the plan's two metrics plus the two that qualify them."""
    return {
        "question_count": len(results),
        "retrieval_hit_rate": round(compute_hit_rate(results), SCORE_PRECISION),
        "retrieval_rank1_hit_rate": round(compute_rank1_hit_rate(results), SCORE_PRECISION),
        "avg_top_score": round(statistics.mean(collect_top_scores(results)), SCORE_PRECISION),
        "top_score": summarize_top_scores(results),
        "rank1_misses": describe_rank1_misses(results),
    }


def describe_rank1_misses(results: list[EvalResult]) -> list[dict]:
    """Name the questions whose expected source was not ranked first.

    A rate on its own says how many failed but not which, and "which" is the only part
    that can be acted on: the three that fail here point at concrete retrieval problems,
    while the rate 0.88 points at nothing. ``ranked_first`` is the document that beat the
    expected one, and ``expected_rank`` is null when it was not returned at all.

    Questions that returned nothing are listed too, with both other fields null: they
    missed rank 1 by the widest margin there is, and leaving them out would make this list
    shorter than the failures the rates count.
    """
    return [
        {
            "question": result.question.question,
            "expected_source": result.question.expected_source,
            "expected_rank": result.expected_rank,
            "ranked_first": result.retrieved[0].title if result.retrieved else None,
        }
        for result in results
        if result.expected_rank != 1
    ]


def evaluate(
    service: RetrievalService, questions: list[EvalQuestion], top_k: int
) -> list[EvalResult]:
    """Retrieve for every question, in order, keeping the results in the same order.

    Retrieval goes through ``RetrievalService`` rather than the repository so the run
    measures the component that actually serves requests, validation included.
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

    # Checked before the model loads: encoding 25 questions is cheap, loading several
    # hundred MB first and then raising is not.
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
