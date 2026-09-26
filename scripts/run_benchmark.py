"""Read-only baseline runner; export provenance spans without redistributing papers.

    python -m scripts.run_benchmark --split test --output eval/runs/baseline-test.json
    python -m scripts.benchmark --predictions eval/runs/baseline-test.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from scripts.benchmark import ROOT, digest, load_benchmark, normalize, score


def run(split: str, k: int, max_words: int, verify_vectors: bool) -> dict:
    # Benchmarking should use cached weights; never silently fetch a different revision.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sqlalchemy import select, text
    import numpy as np
    from app.core.database import SessionLocal
    from app.core.config import get_settings
    from app.models.tables import Chunk, Document
    from app.rag.chunking import TextChunker
    from app.rag.embeddings import get_embedding_client
    from app.services.retrieval import RetrievalService, MAX_TOP_K

    if not 1 <= k <= MAX_TOP_K or max_words < 1:
        raise ValueError(f"Baseline requires 1 <= k <= {MAX_TOP_K} and positive max_words")
    questions, corpus, manifest = load_benchmark()
    selected = [q for q in questions if split == "all" or q["split"] == split]
    by_title = {p["title"]: p for p in manifest["papers"]}
    with SessionLocal() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        docs = list(session.scalars(select(Document)))
        if len(docs) != len(by_title) or {d.title for d in docs} != set(by_title):
            raise ValueError("Database must contain exactly the 20 benchmark documents; use a separate evaluation database")
        id_to_source = {}
        for doc in docs:
            p = by_title[doc.title]
            raw = (ROOT / "eval/corpus" / p["file"]).read_text(encoding="utf-8")
            if doc.content_hash != digest(raw):
                raise ValueError(f"Database document differs from pinned corpus: {doc.title}")
            id_to_source[doc.id] = p["source_id"]
        chunks = list(session.scalars(select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)))
        # Baseline-specific validation. Future systems use the external prediction format
        # and may change chunking, architecture and candidate count freely.
        offsets = {}
        expected_counts = {}
        for source, original in corpus.items():
            expected_counts[source] = len(TextChunker().chunk(original))
        actual_counts = {source: 0 for source in corpus}
        expected_chunks = {source: TextChunker().chunk(original) for source, original in corpus.items()}
        for chunk in chunks:
            source = id_to_source[chunk.document_id]
            actual_counts[source] += 1
            if chunk.embedding is None:
                raise ValueError("Baseline index has missing embeddings")
            if not 0 <= chunk.chunk_index < len(expected_chunks[source]) or normalize(chunk.text) != expected_chunks[source][chunk.chunk_index].text:
                raise ValueError("Database does not use the v0.1.0 chunking configuration")
            original = corpus[source]
            start = sum(len(word) + 1 for word in original.split()[:chunk.chunk_index * 150])
            end = start + len(normalize(chunk.text))
            if original[start:end] != normalize(chunk.text):
                raise ValueError("Chunk provenance mismatch")
            offsets[chunk.id] = {"source_id": source, "start": start, "end": end}
        if actual_counts != expected_counts:
            raise ValueError("Incomplete or duplicate baseline chunks")
        client = get_embedding_client()
        model_hash = hashlib.sha256()
        for key, tensor in sorted(client.model.state_dict().items()):
            model_hash.update(key.encode())
            model_hash.update(tensor.detach().cpu().numpy().tobytes())
        max_error = None
        if verify_vectors:
            vectors = np.asarray(client.embed_texts([c.text for c in chunks]), dtype=np.float32)
            stored = np.asarray([c.embedding for c in chunks], dtype=np.float32)
            max_error = float(np.max(np.abs(vectors - stored)))
            if max_error > 1e-4:
                raise ValueError(f"Stored embeddings do not match the current model: max error {max_error}")
        service = RetrievalService(session, client)
        service.retrieve("benchmark warmup", k)
        predictions = []
        for q in selected:
            started = time.perf_counter()
            retrieved = service.retrieve(q["question"], k)
            elapsed = (time.perf_counter() - started) * 1000
            predictions.append({"id": q["id"], "contexts": [dict(offsets[c.chunk_id], score=c.score) for c in retrieved],
                                "latency_ms": elapsed})
        pg_version = session.scalar(text("SHOW server_version"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    unchanged = subprocess.run(["git", "diff", "--quiet", manifest["baseline_commit"], "--", "app"], cwd=ROOT).returncode == 0
    pipeline_hashes = {p.relative_to(ROOT).as_posix(): digest(p.read_text(encoding="utf-8"))
                       for p in sorted((ROOT / "app").rglob("*.py"))}
    metadata = {
        "benchmark_version": manifest["version"], "benchmark_sha256": manifest["questions_sha256"],
        "corpus_sha256": manifest["corpus_sha256"], "system_commit": commit,
        "pipeline_sha256": digest(json.dumps(pipeline_hashes, sort_keys=True)),
        "runner_sha256": digest(Path(__file__).read_text(encoding="utf-8")),
        "scorer_sha256": digest((ROOT / "scripts/benchmark.py").read_text(encoding="utf-8")),
        "system_name": "dense-pgvector-v0.1.0" if unchanged else "working-tree-retrieval", "mode": "retrieval_only",
        "app_matches_baseline": unchanged,
        "embedding_model": get_settings().embedding_model, "model_weights_sha256": model_hash.hexdigest(),
        "model_max_seq_length": client.model.max_seq_length,
        "model_tokenizer_sha256": digest(client.model.tokenizer.backend_tokenizer.to_str()),
        "stored_vector_max_abs_error": max_error, "vectors_verified": verify_vectors,
        "chunk_size_words": 180, "chunk_overlap_words": 30, "chunk_count": len(chunks),
        "split": split, "k": k, "max_context_words": max_words,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "postgresql": pg_version,
        "dependencies": {p: version(p) for p in ("torch", "sentence-transformers", "transformers", "pgvector")},
        "latency_scope": "single warm run; embedding + database retrieval; excludes model loading, scoring and generation",
        "answerability_note": "No answer/abstain decisions: confusion matrix deliberately unavailable, not zero.",
    }
    return {"metadata": metadata, "predictions": predictions,
            "report": score(selected, predictions, corpus, k, max_words)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="test")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=900)
    parser.add_argument("--skip-vector-verification", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.split, args.k, args.max_words, not args.skip_vector_verification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "question_count": result["report"]["question_count"],
                      "retrieval": result["report"]["retrieval"]}, indent=2))


if __name__ == "__main__":
    main()
