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
    AskRequest,
    AskResponse,
    DocumentCreateRequest,
    DocumentCreateResponse,
    DocumentListItem,
    SourceItem,
)
from app.core.database import get_db_session
from app.rag.embeddings import EmbeddingClient, get_embedding_client
from app.rag.llm import LLMClient, LLMError, get_llm_client
from app.repositories.documents import DocumentRepository
from app.services.answering import AnswerService
from app.services.indexing import IndexingService
from app.services.ingestion import DocumentIngestionService
from app.services.retrieval import RetrievalService

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


@router.post("/ask")
def ask_question(
    payload: AskRequest,
    session: Annotated[Session, Depends(get_db_session)],
    # Three dependencies, three lifetimes. The session is per request; the two clients are
    # process-wide singletons, because building either one would load a second copy of a
    # model -- an embedding model locally, or a connection pool per request for the LLM.
    embedding_client: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
) -> AskResponse:
    """Answer a question from the corpus, with the sources the answer rests on.

    Read-only. Retrieval reads, generation is a remote call, and nothing is written here;
    recording the question is a separate concern with its own failure modes (Task 12).

    An empty corpus is not an error: retrieval returns no chunks, the service answers
    "the sources do not contain enough information" without calling the model at all, and
    this returns 200. A question the corpus cannot answer is a normal outcome.
    """
    service = AnswerService(RetrievalService(session, embedding_client), llm_client)

    try:
        # payload.top_k always has a value -- the schema fills in DEFAULT_TOP_K -- so the
        # service's own default applies only to callers that are not HTTP.
        result = service.answer(payload.question, payload.top_k)
    except ValueError as exc:
        # Same mapping as POST /documents: the service raises ValueError because it knows
        # nothing about HTTP, and 422 keeps "malformed field" and "unacceptable value"
        # indistinguishable to a client that has no use for the distinction.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except LLMError as exc:
        # 502: the request was well formed, this code did its part, and the endpoint we
        # depend on failed. Not the caller's fault, and the same body may well succeed.
        #
        # One status for every upstream failure -- unreachable host, timeout, 401, 429 --
        # because the caller's next move is the same in all of them. A timeout arguably
        # deserves 504; that is worth splitting only once something acts on the difference.
        #
        # The message is the chained exception's, which carries the provider's status but
        # never the API key: the key travels in a header, and httpx does not put headers in
        # its exception text.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return AskResponse(
        answer=result.answer,
        sources=[SourceItem.model_validate(source) for source in result.sources],
    )
