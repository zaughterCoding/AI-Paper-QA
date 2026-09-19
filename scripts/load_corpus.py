"""Load the corpus in eval/corpus/ into the database through the HTTP API.

Start the service in another terminal first, then run:

    python -m uvicorn app.main:app --port 8000
    python scripts\\load_corpus.py

Going through HTTP instead of calling IngestService directly means the real
path is exercised: routing, dependency injection and the real embedding client,
which the API tests replace with a fake. That is the only way the
"import indexes immediately" path gets run against the real model.

Idempotent: POST /documents dedupes on content_hash, so re-importing a paper
returns created=false and embedded_chunk_count=0.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).resolve().parents[1] / "eval" / "corpus"


def load_manifest() -> list[dict]:
    path = CORPUS_DIR / "sources.json"
    if not path.exists():
        raise SystemExit(f"Not found: {path}. Run scripts/fetch_corpus.py first to fetch the corpus")
    return json.loads(path.read_text(encoding="utf-8"))["papers"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Load the corpus into the database")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    papers = load_manifest()

    # Generous timeout: the first request loads the embedding model (hundreds of MB,
    # seconds to tens of seconds) and each paper encodes hundreds of chunks, so the
    # default 5s read timeout would fire on the very first paper.
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=60.0)

    failures = 0
    with httpx.Client(base_url=args.base_url, timeout=timeout) as client:
        for paper in papers:
            path = CORPUS_DIR / paper["file"]
            payload = {
                "title": paper["title"],
                "source": paper["url"],
                "content": path.read_text(encoding="utf-8"),
            }

            try:
                response = client.post("/documents", json=payload)
            except httpx.HTTPError as error:
                print(f"  failed  {paper['slug']:32} {type(error).__name__}: {error}")
                failures += 1
                continue

            if response.status_code != 201:
                # Include the body: a 422 carries the actual validation error, which
                # a bare status code would leave to a manual reproduction.
                print(f"  failed  {paper['slug']:32} HTTP {response.status_code}  {response.text[:200]}")
                failures += 1
                continue

            body = response.json()
            state = "new" if body["created"] else "existing"
            print(
                f"  {state}  {paper['slug']:32} "
                f"{body['chunk_count']:>4} chunks, embedded now: {body['embedded_chunk_count']:>4}"
            )

    print()
    if failures:
        print(f"⚠️  {failures} papers failed to import")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
