"""Text chunking tests.

Chunking is the first step of the pipeline: a wrong split silently degrades everything
downstream, without raising anything, so the boundaries are covered closely.
"""

import pytest

from app.rag.chunking import TextChunker


def test_chunker_returns_empty_list_for_empty_text():
    """Empty text yields no chunks, not an exception and not an empty chunk."""
    assert TextChunker().chunk("") == []


def test_chunker_returns_empty_list_for_whitespace_only_text():
    """Whitespace-only text counts as empty."""
    assert TextChunker().chunk("   \n\t  ") == []


def test_chunker_splits_text_with_overlap():
    """The main splitting case, with chunk_size=4 and overlap=1."""
    text = " ".join(str(i) for i in range(10))  # "0 1 2 3 4 5 6 7 8 9"
    chunks = TextChunker(chunk_size=4, overlap=1).chunk(text)

    assert [chunk.text for chunk in chunks] == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_chunker_returns_single_chunk_when_text_fits():
    """Text shorter than chunk_size produces one chunk."""
    chunks = TextChunker(chunk_size=100, overlap=10).chunk("hello world")

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunk_indices_are_sequential_and_start_at_zero():
    """Indices are 0,1,2,...; the unique constraint on (document_id, chunk_index)
    depends on it."""
    text = " ".join(str(i) for i in range(20))
    chunks = TextChunker(chunk_size=4, overlap=1).chunk(text)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_token_count_matches_word_count_of_chunk_text():
    """token_count stays consistent with the words in text."""
    text = " ".join(str(i) for i in range(20))
    chunks = TextChunker(chunk_size=5, overlap=2).chunk(text)

    for chunk in chunks:
        assert chunk.token_count == len(chunk.text.split())


def test_every_word_is_covered_by_some_chunk():
    """Invariant: every word of the input appears in at least one chunk."""
    # Prime length, so an exact division cannot hide an off-by-one at the edge.
    words = [f"w{i}" for i in range(37)]
    chunks = TextChunker(chunk_size=8, overlap=3).chunk(" ".join(words))

    covered = {word for chunk in chunks for word in chunk.text.split()}
    assert covered == set(words)


def test_chunking_is_deterministic():
    """Same input, same output; nothing else can be tested or evaluated otherwise."""
    text = " ".join(str(i) for i in range(50))
    chunker = TextChunker(chunk_size=10, overlap=3)

    assert [c.text for c in chunker.chunk(text)] == [c.text for c in chunker.chunk(text)]


def test_chunker_rejects_overlap_not_smaller_than_chunk_size():
    """overlap >= chunk_size would never advance and must be rejected."""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, overlap=15)


def test_chunker_rejects_non_positive_chunk_size():
    """chunk_size <= 0 must fail at construction rather than produce garbage.

    Both cases use overlap < chunk_size, so they slip past the other check and exercise
    this guard on its own.
    """
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, overlap=-1)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=-5, overlap=-10)
