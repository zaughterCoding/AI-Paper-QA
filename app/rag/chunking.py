"""Split long documents into overlapping chunks.

Chunking is necessary because an embedding model has a maximum input length, and
because retrieval needs focused text: a chunk covering five topics cannot tell you
which one matched.

Neighbouring chunks overlap so that a sentence crossing a boundary survives intact in
at least one of them. Without overlap, "...the problem of long-range" and "dependencies
by relating all positions..." are each only half a concept, and neither is retrievable.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """One chunk of text. Frozen, because a chunk is never modified once split."""

    index: int  # position within the parent document, from 0
    text: str
    token_count: int  # approximate, see TextChunker


class TextChunker:
    """Split text on whitespace, keeping ``overlap`` words between neighbours.

    ``token_count`` is approximated by word count. Running the model's real tokeniser is
    expensive, and the number only gauges chunk size, never feeding a calculation that
    has to be exact.
    """

    def __init__(self, chunk_size: int = 180, overlap: int = 30) -> None:
        # Validated here rather than inside chunk(): a bad configuration should fail
        # immediately and loudly, not after a few hundred documents.
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must not be negative")
        if overlap >= chunk_size:
            # Otherwise the slice start would never advance: the next start is
            # end - overlap <= start, and the loop would spin in place.
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        """Split ``text``. Blank or whitespace-only input yields an empty list."""
        # A bare split() splits on any whitespace and drops empty strings, so
        # "   \n\t " and "" take the same path.
        words = text.split()
        if not words:
            return []

        chunks: list[TextChunk] = []
        start = 0
        index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]

            chunks.append(
                TextChunk(
                    index=index,
                    text=" ".join(chunk_words),
                    token_count=len(chunk_words),
                )
            )

            # Without this, the final chunk would be emitted once more, fully contained
            # in the one before it.
            if end == len(words):
                break

            # Step back by overlap. Since overlap < chunk_size, start strictly increases
            # and the loop terminates.
            start = end - self.overlap
            index += 1

        return chunks
