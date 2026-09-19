"""Embedding client tests.

Contract tests run the same assertions against the fake client and the real one, so the
fake cannot drift. Real-model tests are skipped by default and run with
PAPERQA_RUN_MODEL_TESTS=1.
"""

import math
import os
from functools import lru_cache

import pytest

from app.core.config import EMBEDDING_DIM, get_settings
from app.rag.embeddings import EmbeddingClient
from tests.fakes import FakeEmbeddingClient

RUNS_REAL_MODEL = os.environ.get("PAPERQA_RUN_MODEL_TESTS") == "1"
SKIP_REASON = "real-model tests are off by default; set PAPERQA_RUN_MODEL_TESTS=1 to enable"


@lru_cache(maxsize=1)
def _real_client() -> EmbeddingClient:
    """Load the real model once per session; it takes seconds and hundreds of MB."""
    return EmbeddingClient(get_settings().embedding_model)


def _make_fake() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


def _make_real() -> EmbeddingClient:
    if not RUNS_REAL_MODEL:
        pytest.skip(SKIP_REASON)
    return _real_client()


@pytest.fixture(params=[pytest.param(_make_fake, id="fake"), pytest.param(_make_real, id="real")])
def client(request: pytest.FixtureRequest):
    """Runs each test twice: once on the fake client, once on the real model."""
    return request.param()


@pytest.fixture(scope="session")
def real_client() -> EmbeddingClient:
    if not RUNS_REAL_MODEL:
        pytest.skip(SKIP_REASON)
    return _real_client()


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; both vectors are normalized, so the dot product is enough."""
    return sum(x * y for x, y in zip(a, b))


# --- contract tests: both the fake and the real client must satisfy these ---


def test_embed_text_returns_vector_of_schema_dimension(client) -> None:
    """The dimension must match the schema, where chunks.embedding is vector(384).

    Binding the component to the schema makes a model change fail here rather than as an
    obscure pgvector type error on insert.
    """
    vector = client.embed_text("self attention mechanism")

    assert isinstance(vector, list)
    assert len(vector) == EMBEDDING_DIM
    assert all(isinstance(value, float) for value in vector)


def test_vectors_are_normalized(client) -> None:
    """A normalized vector has length 1.

    pgvector's cosine distance `<=>` and inner product `<#>` agree on unit vectors, and
    normalization removes the length bias that would otherwise skew the ranking.
    """
    vector = client.embed_text("self attention mechanism")

    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embed_texts_returns_one_vector_per_text_in_order(client) -> None:
    vectors = client.embed_texts(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    # approx rather than ==: padding during batch encoding can move the last few
    # floating point digits compared with encoding a single text
    assert vectors[0] == pytest.approx(client.embed_text("alpha"), abs=1e-6)
    assert vectors[2] == pytest.approx(client.embed_text("gamma"), abs=1e-6)


def test_embed_texts_of_empty_list_returns_empty(client) -> None:
    """An empty list must never reach the model.

    The early return is not just about time: some model versions raise on empty input,
    while having no text is a perfectly normal state (a document with no chunks).
    """
    assert client.embed_texts([]) == []


def test_same_text_gives_same_vector(client) -> None:
    assert client.embed_text("deterministic") == client.embed_text("deterministic")


# --- the fake client's own contract ---


def test_fake_client_similarity_follows_shared_words() -> None:
    """Pins the fake's limits so nobody mistakes it for a semantic model.

    It matches shared vocabulary only, so it can test that ranking works, never whether
    retrieval is accurate.
    """
    fake = FakeEmbeddingClient()

    base = fake.embed_text("self attention mechanism")
    related = fake.embed_text("self attention layer")
    unrelated = fake.embed_text("banana bread recipe")

    assert _cosine(base, related) > _cosine(base, unrelated)


# --- real model only: it understands meaning ---


def test_real_model_places_related_words_closer(real_client: EmbeddingClient) -> None:
    """The real model's whole value: similar meaning, similar vector."""
    cat = real_client.embed_text("cat")
    kitten = real_client.embed_text("kitten")
    airplane = real_client.embed_text("airplane")

    assert _cosine(cat, kitten) > _cosine(cat, airplane)


def test_real_model_matches_paragraph_when_words_overlap(
    real_client: EmbeddingClient,
) -> None:
    """With shared vocabulary, the relevant paragraph must rank first."""
    question = real_client.embed_text("What is self-attention?")
    relevant = real_client.embed_text(
        "Self-attention relates all positions in a sequence to compute its representation."
    )
    irrelevant = real_client.embed_text("We trained the model for three days on eight GPUs.")

    assert _cosine(question, relevant) > _cosine(question, irrelevant)


def test_real_model_matches_chunk_when_question_is_paraphrased(
    real_client: EmbeddingClient,
) -> None:
    """The key case: a paraphrased question with no shared keywords still matches.

    What retrieval actually compares is chunks, not single sentences. Measuring on single
    sentences underestimates the system badly, because one shared word is enough to flip
    the ranking.
    """
    question = real_client.embed_text("How does the model handle long-range dependencies?")
    attention_chunk = real_client.embed_text(
        "The Transformer follows an encoder-decoder structure using stacked self-attention "
        "and point-wise, fully connected layers. Self-attention relates the different positions "
        "of a single sequence in order to compute a representation of that sequence. This allows "
        "the model to capture dependencies between distant positions regardless of their distance "
        "in the sequence, unlike recurrent networks which must process the input sequentially."
    )
    training_chunk = real_client.embed_text(
        "We trained the base models for a total of 100,000 steps or 12 hours on eight NVIDIA P100 "
        "GPUs. We used the Adam optimizer with a custom learning rate schedule that increases "
        "linearly for the first 4000 steps and then decays proportionally to the inverse square "
        "root of the step number. Dropout was applied to the output of each sub-layer."
    )

    assert _cosine(question, attention_chunk) > _cosine(question, training_chunk)


def test_real_model_gives_irrelevant_chunk_a_low_score(
    real_client: EmbeddingClient,
) -> None:
    """An unrelated chunk must score low, which is what makes a similarity threshold
    possible. The 0.3 ceiling only has to catch a degenerate model that scores
    everything high.
    """
    question = real_client.embed_text("How does the model handle long-range dependencies?")
    unrelated = real_client.embed_text(
        "We trained on the standard WMT 2014 English-German dataset consisting of about "
        "4.5 million sentence pairs, encoded using byte-pair encoding."
    )

    assert _cosine(question, unrelated) < 0.3


def test_known_limitation_single_sentence_with_keyword_sharing_distractor(
    real_client: EmbeddingClient,
) -> None:
    """Known limitation: with single-sentence candidates, a distractor sharing one common
    word wins over a related sentence that shares no content word.

    This is the inherent boundary of vocabulary-driven retrieval on a small 384-dimension
    model, not a bug in any one model. The chunk-level test above shows the real system is
    not affected, so this marks where the boundary lies rather than raising an alarm.
    """
    question = real_client.embed_text("How does the model handle long-range dependencies?")
    related_but_no_shared_content_word = real_client.embed_text(
        "Self-attention relates all positions in a sequence to compute its representation."
    )
    unrelated_but_shares_the_word_model = real_client.embed_text(
        "We trained the model for three days on eight GPUs."
    )

    assert _cosine(question, unrelated_but_shares_the_word_model) > _cosine(
        question, related_but_no_shared_content_word
    )
