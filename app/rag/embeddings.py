"""Turn text into vectors."""

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingClient:
    """A thin wrapper around sentence-transformers.

    Wrapping it keeps the choice of implementation in one file, lets tests substitute a
    fake exposing the same two methods, and fixes decisions such as whether vectors are
    normalised so that no caller decides differently.

    The model is loaded in ``__init__``, which takes seconds and hundreds of megabytes.
    Callers must therefore reuse a single instance rather than constructing one per
    request -- see ``get_embedding_client``.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string."""
        # Delegating keeps one code path. Two would eventually drift, for example by
        # normalising in one and not the other.
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        if not texts:
            # Return early rather than handing an empty batch to the model, which some
            # versions reject. "No texts" is a normal case, not an error.
            return []

        vectors = self.model.encode(texts, normalize_embeddings=True)
        # Plain Python floats, not numpy scalars: callers should not need numpy, and
        # numpy's float32 does not serialise to JSON.
        return [vector.tolist() for vector in vectors]


# Process-wide instance. The leading underscore marks it as internal state.
_default_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """Return the process-wide EmbeddingClient, creating it on first use.

    It must be a singleton: every instance loads the model, so constructing one per
    request would exhaust memory on the first burst of concurrency.

    It is a plain zero-argument function, so FastAPI's ``Depends`` accepts it and
    ``rag/`` stays free of any web-framework import. Loading is lazy, so collecting
    tests never pays for the model.
    """
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient(get_settings().embedding_model)
    return _default_client
