"""Reads and writes for the ``documents`` table."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.tables import Document


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        # The repository holds the session but does not open, commit or close it;
        # transaction boundaries belong to the service layer.
        self.session = session

    def create_document(self, title: str, source: str, content_hash: str) -> Document:
        """Insert a document and return it with its database-generated fields."""
        document = Document(title=title, source=source, content_hash=content_hash)
        self.session.add(document)

        # flush is required: the column default uuid.uuid4 is applied when the INSERT is
        # sent, so without it document.id is still None and cannot serve as the chunks'
        # foreign key. Note that flush is not commit -- the SQL goes out, the transaction
        # stays open and can still be rolled back.
        self.session.flush()
        return document

    def list_documents(self) -> list[Document]:
        """Every document, newest first."""
        # scalars() returns a one-shot iterator; wrapping it in list() stops callers from
        # receiving an object that is empty after a single pass.
        return list(
            self.session.scalars(select(Document).order_by(Document.created_at.desc()))
        )

    def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        """Look up a document by primary key."""
        # session.get() consults the identity map first and skips the query on a hit,
        # which is the recommended way to fetch by primary key.
        return self.session.get(Document, document_id)

    def get_by_content_hash(self, content_hash: str) -> Document | None:
        """Look up a document by content fingerprint, to detect a repeat import."""
        # Hits the unique index rather than scanning the table. Returns None when there is
        # no match, the standard way to express "may be absent".
        return self.session.scalar(
            select(Document).where(Document.content_hash == content_hash)
        )
