"""Backfill embeddings for every chunk in the database that has none.

    python scripts\\index_pending.py

One-shot script, not a service: it opens its own session and exits when done.
Useful for documents imported before indexing was wired into POST /documents,
and for imports whose indexing step failed.

Only chunks with a NULL embedding are processed; existing vectors are never
overwritten, so re-encoding after changing the embedding model would need a
separate overwrite pass. Mixing vectors from two models in one index would
degrade retrieval rather than fail.
"""

import sys

from app.core.database import SessionLocal
from app.rag.embeddings import get_embedding_client
from app.services.indexing import IndexingService


def main() -> int:
    session = SessionLocal()
    try:
        # This line loads the model (seconds, hundreds of MB) -- acceptable for a script.
        service = IndexingService(session, get_embedding_client())
        result = service.index_all_pending()
    finally:
        session.close()

    print(f"Embedded {result.embedded_count} chunks, skipped {result.skipped_count} (already had vectors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
