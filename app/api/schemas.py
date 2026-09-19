"""Request and response models for the HTTP layer.

These are deliberately separate from the database models and share no code with them. The
ORM models describe how data is stored -- primary keys, foreign keys, constraints,
cascades -- while these describe what the API accepts and returns, which is only the
fields a caller needs.

Coupling them looks like a saving but causes two immediate problems: an internal column
such as ``embedding`` would become part of the public response, and changing a table would
silently change the API contract. So the boundary converts explicitly: request -> schema
-> service -> ORM -> schema -> response.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreateRequest(BaseModel):
    """Body of POST /documents.

    These Field constraints are the first gate: a wrong type, a missing field or an
    out-of-range length is rejected by pydantic with a 422 before any business code runs.
    They can only express shape, not rules such as "the title must not be all whitespace",
    which the service enforces.
    """

    title: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)


class DocumentCreateResponse(BaseModel):
    document_id: UUID
    chunk_count: int
    # False when the same content had already been imported, so the caller can tell
    # "this really was stored" from "this was already here".
    created: bool
    # How many chunks had their vector written by this request.
    #
    # Reporting it matters because ingestion and indexing are separate steps: with only
    # chunk_count, a caller could not tell whether indexing happened at all, and "the
    # document was stored but every vector is NULL" would look identical to success. The
    # three possible values are:
    #   == chunk_count          -> new document, every chunk embedded
    #   == 0 and created=False  -> repeat import of an already-indexed document
    #   >  0 and created=False  -> repeat import that filled in previously missing vectors
    embedded_chunk_count: int


class DocumentListItem(BaseModel):
    """One entry in the GET /documents response.

    ``from_attributes=True`` lets pydantic read ORM attributes directly instead of going
    through a hand-written dict conversion; without it, model_validate(orm_object) fails,
    because pydantic expects a mapping by default.

    ``content_hash`` is deliberately absent: it is an internal deduplication fingerprint
    with no meaning for a client.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    source: str
    created_at: datetime
