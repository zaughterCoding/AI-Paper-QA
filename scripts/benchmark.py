"""Frozen evidence benchmark. Pure scoring needs only the Python standard library."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "eval/benchmark/v1"
METRIC_VERSION = "evidence-coverage-v1"


def normalize(text: str) -> str:
    return " ".join(text.split())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_benchmark(directory: Path = BENCHMARK) -> tuple[list[dict], dict[str, str], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    question_text = (directory / "questions.jsonl").read_text(encoding="utf-8")
    if digest(question_text) != manifest["questions_sha256"]:
        raise ValueError("Frozen questions changed; create a new benchmark version, do not overwrite v1")
    questions = read_jsonl(directory / "questions.jsonl")
    corpus = {}
    for paper in manifest["papers"]:
        path = ROOT / "eval/corpus" / paper["file"]
        text = normalize(path.read_text(encoding="utf-8"))
        if digest(text) != paper["normalized_sha256"]:
            raise ValueError(f"Corpus drift: {paper['source_id']}; restore the pinned corpus before comparing")
        corpus[paper["source_id"]] = text
    actual_corpus_hash = digest(json.dumps({source: digest(text) for source, text in corpus.items()}, sort_keys=True))
    if actual_corpus_hash != manifest["corpus_sha256"]:
        raise ValueError("Manifest corpus fingerprint is inconsistent")
    ids, families, seen_questions, fact_splits = set(), {}, set(), {}
    for q in questions:
        if q["id"] in ids or normalize(q["question"]) in seen_questions:
            raise ValueError("Duplicate benchmark ID or question")
        ids.add(q["id"])
        seen_questions.add(normalize(q["question"]))
        if type(q["answerable"]) is not bool or q["split"] not in {"dev", "test"}:
            raise ValueError(f"Invalid answerability/split: {q['id']}")
        previous = families.setdefault(q["family"], q["split"])
        if previous != q["split"]:
            raise ValueError(f"Paraphrase family crosses splits: {q['family']}")
        if not q["question"].strip() or not q["reference_answer"].strip():
            raise ValueError(f"Missing question/reference: {q['id']}")
        groups = q["evidence_groups"]
        if bool(groups) != q["answerable"]:
            raise ValueError(f"Evidence/answerability mismatch: {q['id']}")
        if not q["answerable"] and not q.get("unanswerable_reason"):
            raise ValueError(f"Negative needs a rationale: {q['id']}")
        for group in groups:
            if not group:
                raise ValueError("Empty evidence group")
            for e in group:
                text = corpus[e["source_id"]]
                start, end = e["start"], e["end"]
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                    raise ValueError(f"Invalid evidence offsets: {q['id']}")
                if text[start:end] != e["quote"]:
                    raise ValueError(f"Evidence does not match original text: {q['id']}")
                key = (e["source_id"], start, end)
                if fact_splits.setdefault(key, q["split"]) != q["split"]:
                    raise ValueError(f"Component evidence crosses dev/test splits: {q['id']}")
    if len(questions) != manifest["question_count"]:
        raise ValueError("Question count changed")
    return questions, corpus, manifest


def context_span(context: dict, corpus: dict[str, str], remaining_words: int) -> tuple[str, int, int, int]:
    source = context["source_id"]
    if source not in corpus:
        raise ValueError(f"Unknown source: {source}")
    original = corpus[source]
    if "start" in context and "end" in context:
        start, end = context["start"], context["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(original):
            raise ValueError("Invalid context offsets")
        text = original[start:end]
        if "text" in context and normalize(context["text"]) != normalize(text):
            raise ValueError("Context text disagrees with offsets")
    else:
        text = normalize(context.get("text", ""))
        start = original.find(text)
        if not text or start < 0:
            raise ValueError("Context is not a verbatim source passage; export its original evidence spans")
        if original.find(text, start + 1) >= 0:
            raise ValueError("Ambiguous repeated passage; provide explicit start/end offsets")
    start += len(text) - len(text.lstrip())
    text = text.strip()
    if not text:
        raise ValueError("Context must contain non-whitespace text")
    words = text.split()
    used = min(len(words), remaining_words)
    end = start + len(" ".join(words[:used]))
    return source, start, end, used


def covered(evidence: dict, spans: list[tuple[int, int]], text: str) -> bool:
    cursor = evidence["start"]
    for start, end in sorted(spans):
        if end <= cursor:
            continue
        if start > cursor and text[cursor:start].strip():
            return False
        cursor = max(cursor, end)
        if cursor >= evidence["end"]:
            return True
    return False


def mean(values: list) -> float | None:
    return statistics.mean(values) if values else None


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[dict]) -> dict:
    positives = [r for r in rows if r["answerable"]]
    retrieval = {
        "positive_count": len(positives),
        "evidence_hit_rate": mean([r["evidence_recall"] > 0 for r in positives]),
        "evidence_recall": mean([r["evidence_recall"] for r in positives]),
        "complete_evidence_rate": mean([r["evidence_recall"] == 1 for r in positives]),
        "mrr": mean([r["reciprocal_rank"] for r in positives]),
        "mean_context_words": mean([r["context_words"] for r in rows]),
    }
    answerability = None
    if rows and all("answered" in r for r in rows):
        matrix = {key: 0 for key in ("TP", "FP", "TN", "FN")}
        for r in rows:
            key = ("TP" if r["answerable"] else "FP") if r["answered"] else ("FN" if r["answerable"] else "TN")
            matrix[key] += 1
        tp, fp, tn, fn = (matrix[key] for key in ("TP", "FP", "TN", "FN"))
        answerability = {"confusion_matrix": matrix,
                         "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                         "false_positive_rate": ratio(fp, fp + tn),
                         "false_negative_rate": ratio(fn, fn + tp),
                         "accuracy": ratio(tp + tn, len(rows))}
    judged = [r["answer_correct"] for r in positives if r.get("answer_correct") is not None]
    latencies = sorted(r["latency_ms"] for r in rows if "latency_ms" in r)
    citation_rows = [r for r in rows if "citation_count" in r]
    claim_rows = [r for r in rows if "claim_count" in r]
    return {"question_count": len(rows), "retrieval": retrieval, "answerability": answerability,
            "generation": {"judged_answer_count": len(judged), "answer_accuracy": mean(judged),
                           "judgment_coverage": ratio(len(judged), len(positives)),
                           "judged_citation_question_count": len(citation_rows),
                           "citation_precision": ratio(sum(r["citation_supported"] for r in citation_rows), sum(r["citation_count"] for r in citation_rows)),
                           "judged_claim_question_count": len(claim_rows),
                           "claim_support_rate": ratio(sum(r["claims_supported"] for r in claim_rows), sum(r["claim_count"] for r in claim_rows))},
            "latency": {"measured_count": len(latencies), "p50_ms": statistics.median(latencies) if latencies else None,
                        "p95_ms": latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None}}


def score(questions: list[dict], predictions: list[dict], corpus: dict[str, str],
          k: int = 5, max_words: int = 900) -> dict:
    if type(k) is not int or k < 1 or type(max_words) is not int or max_words < 1:
        raise ValueError("k and max_words must be positive integers")
    expected = {q["id"] for q in questions}
    actual = [p["id"] for p in predictions]
    if len(expected) != len(questions) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Predictions must contain exactly one row for every selected question")
    decisions = ["answered" in p for p in predictions]
    if any(decisions) and not all(decisions):
        raise ValueError("Provide answered for all selected questions, or none")
    indexed = {p["id"]: p for p in predictions}
    rows = []
    for q in questions:
        p = indexed[q["id"]]
        for field in ("answered", "answer_correct"):
            if field in p and (p[field] is not None or field == "answered") and type(p[field]) is not bool:
                raise ValueError(f"{field} must be boolean")
        if p.get("answer_correct") is not None and "answered" not in p:
            raise ValueError("Answer correctness requires an explicit answered decision")
        if p.get("answer_correct") is True and p.get("answered") is False:
            raise ValueError("An abstention cannot be a correct substantive answer")
        for numerator, denominator in (("citation_supported", "citation_count"), ("claims_supported", "claim_count")):
            if numerator in p or denominator in p:
                if not all(type(p.get(f)) is int for f in (numerator, denominator)):
                    raise ValueError("Judgment counts must be complete integer pairs")
                if not 0 <= p[numerator] <= p[denominator]:
                    raise ValueError("Supported counts must lie between zero and total counts")
                if "answered" not in p:
                    raise ValueError("Generation judgments require explicit answered decisions")
        if "latency_ms" in p:
            latency = p["latency_ms"]
            if type(latency) not in (float, int) or not math.isfinite(latency) or latency < 0:
                raise ValueError("Invalid latency")
        spans = defaultdict(list)
        first_rank, words, hits = None, 0, 0
        contexts = p["contexts"]
        if not isinstance(contexts, list):
            raise ValueError("contexts must be a list")
        if "answered" in p:
            actual_words = sum(context_span(ctx, corpus, 10**12)[3] for ctx in contexts)
            if len(contexts) > k or actual_words > max_words:
                raise ValueError("Generation context exceeds budget; truncate before generating, not afterward")
        for rank, context in enumerate(contexts[:k], 1):
            source, start, end, used = context_span(context, corpus, max_words - words)
            words += used
            if used:
                spans[source].append((start, end))
            hits = sum(any(covered(e, spans[e["source_id"]], corpus[e["source_id"]]) for e in group)
                       for group in q["evidence_groups"])
            if hits and first_rank is None:
                first_rank = rank
            if words >= max_words:
                break
        row = {"id": q["id"], "category": q["category"], "split": q["split"], "answerable": q["answerable"],
               "evidence_recall": hits / len(q["evidence_groups"]) if q["answerable"] else None,
               "reciprocal_rank": 1 / first_rank if first_rank else 0,
               "context_words": words}
        row.update({f: p[f] for f in ("answered", "answer_correct", "latency_ms", "citation_supported", "citation_count", "claims_supported", "claim_count") if f in p})
        rows.append(row)
    result = summarize(rows)
    result.update({"metric_version": METRIC_VERSION, "k": k, "max_context_words": max_words, "per_question": rows})
    for dimension in ("category", "split"):
        result[f"by_{dimension}"] = {value: summarize([r for r in rows if r[dimension] == value])
                                      for value in sorted({r[dimension] for r in rows})}
    tags = sorted({tag for q in questions for tag in q.get("tags", [])})
    result["by_tag"] = {tag: summarize([r for q, r in zip(questions, rows) if tag in q.get("tags", [])]) for tag in tags}
    return result


def compare(current: dict, baseline: dict) -> dict:
    old = {r["id"]: r for r in baseline["per_question"]}
    new = {r["id"]: r for r in current["per_question"]}
    if set(old) != set(new) or any(current[f] != baseline[f] for f in ("k", "max_context_words")):
        raise ValueError("Paired comparison requires the same questions and context budgets")
    positive_ids = [key for key in new if new[key]["answerable"]]
    metrics = ("evidence_hit_rate", "evidence_recall", "complete_evidence_rate", "mrr")
    return {"retrieval_delta": {key: current["retrieval"][key] - baseline["retrieval"][key]
                                if current["retrieval"][key] is not None and baseline["retrieval"][key] is not None else None for key in metrics},
            "improvements": [key for key in positive_ids if new[key]["evidence_recall"] > old[key]["evidence_recall"]],
            "regressions": [key for key in positive_ids if new[key]["evidence_recall"] < old[key]["evidence_recall"]],
            "note": "Paired descriptive differences, not a statistical significance claim."}


def read_run(path: Path, manifest: dict) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    for field, expected in (("benchmark_sha256", manifest["questions_sha256"]), ("corpus_sha256", manifest["corpus_sha256"])):
        if run["metadata"][field] != expected:
            raise ValueError(f"Run uses a different {field}")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="Run JSON containing metadata and predictions")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, help="Rescore and compare a baseline run on the same questions")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="test")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=900)
    args = parser.parse_args()
    if args.compare and not args.predictions:
        parser.error("--compare requires --predictions")
    questions, corpus, manifest = load_benchmark()
    selected = [q for q in questions if args.split == "all" or q["split"] == args.split]
    if args.predictions:
        run = read_run(args.predictions, manifest)
        # Do not filter unknown/duplicate IDs out of the input: those are errors.
        result = score(selected, run["predictions"], corpus, args.k, args.max_words)
        result["metadata"] = run["metadata"]
        if args.compare:
            baseline = read_run(args.compare, manifest)
            result["comparison"] = compare(result, score(selected, baseline["predictions"], corpus, args.k, args.max_words))
    else:
        result = {"benchmark": manifest["version"], "question_count": len(selected),
                  "categories": dict(Counter(q["category"] for q in selected)),
                  "answerability": dict(Counter(str(q["answerable"]) for q in selected)),
                  "corpus_documents": len(corpus), "validation": "passed"}
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
