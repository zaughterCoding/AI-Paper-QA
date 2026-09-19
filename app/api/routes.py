"""HTTP routes.

This layer does three things and no more: parse the request (via the pydantic schemas),
call a service, and translate the result or the exception into HTTP. It writes no SQL and
makes no business decisions. ``session.commit()`` lives in the services, so the absence of
a commit here is a consequence of the layering rather than an omission.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import (
    DocumentCreateRequest,
    DocumentCreateResponse,
    DocumentListItem,
)
from app.core.database import get_db_session
from app.rag.embeddings import EmbeddingClient, get_embedding_client
from app.repositories.documents import DocumentRepository
from app.services.indexing import IndexingService
from app.services.ingestion import DocumentIngestionService

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Liveness probe: is the process up? Deliberately does not touch the database."""
    return {"status": "ok"}


@router.post("/documents", status_code=status.HTTP_201_CREATED)
def create_document(
    payload: DocumentCreateRequest,
    # Annotated[Session, Depends(...)] is the current FastAPI style. The older
    # `session: Session = Depends(...)` also works, but it puts a function call in a
    # default-value position, which Python evaluates once at definition time, and
    # Annotated keeps that information in the type where it belongs.
    session: Annotated[Session, Depends(get_db_session)],
    embedding_client: Annotated[EmbeddingClient, Depends(get_embedding_client)],
) -> DocumentCreateResponse:
    """Import a document and embed its chunks.

    Two steps, two transactions: ingest stores the document and its chunks as text, then
    indexing fills in the vectors. Because they are separate, a failure in the second step
    leaves the document stored -- the request returns 500, but the data is not wasted.
    Resubmitting the same content takes the "already imported" path and runs indexing
    again, which fills in the missing vectors, so the endpoint is self-healing.
    """
    service = DocumentIngestionService(session)

    try:
        result = service.ingest(
            title=payload.title, source=payload.source, content=payload.content
        )
    except ValueError as exc:
        # The service raises ValueError because it knows nothing about HTTP; mapping a
        # business error onto a status code is this layer's job. 422 rather than 400 keeps
        # it consistent with pydantic's validation errors, since the client does not need
        # to distinguish a malformed field from an unacceptable one.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    # Indexing is called explicitly rather than hidden inside ingest, so that "who does
    # the encoding" is visible in the code. It is not wrapped in try/except: an indexing
    # failure should be a 500, never a 201 that looks successful while no vectors exist.
    indexing = IndexingService(session, embedding_client).index_document(result.document_id)

    return DocumentCreateResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        created=result.created,
        embedded_chunk_count=indexing.embedded_count,
    )


@router.get("/documents")
def list_documents(
    session: Annotated[Session, Depends(get_db_session)],
) -> list[DocumentListItem]:
    """Every imported document, newest first."""
    documents = DocumentRepository(session).list_documents()
    return [DocumentListItem.model_validate(document) for document in documents]
