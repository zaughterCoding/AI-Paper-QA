"""Answer service and LLM client tests.

Two things are being tested and they fail differently.

The prompt is the whole product of the answering layer, and the model's reply is a string,
so a prompt missing the question or the sources still returns something that reads like an
answer. Only assertions on the prompt itself catch that, which is why the fake LLM client
records what it was given.

Ordering and call counts are the other half: validation must happen before the model is
asked anything, and with no sources the model must not be asked at all. Both mistakes
produce a normal-looking answer and only a cost or latency signal.

Real API calls are skipped by default and enabled with PAPERQA_RUN_LLM_TESTS=1, matching
how the real embedding model is gated.
"""

import json
import os
import uuid

import httpx
import pytest
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.tables import Chunk, Document
from app.rag import llm
from app.rag.llm import MAX_ANSWER_TOKENS, LLMClient
from app.repositories.chunks import RetrievedChunk
from app.services.answering import (
    NO_SOURCES_ANSWER,
    AnswerResult,
    AnswerService,
    build_prompt,
    format_sources,
)
from app.services.retrieval import DEFAULT_TOP_K, MAX_TOP_K, RetrievalService
from tests.fakes import FakeEmbeddingClient, FakeLLMClient, FakeRetrievalService

RUNS_REAL_LLM = os.environ.get("PAPERQA_RUN_LLM_TESTS") == "1"
SKIP_REASON = "real-LLM tests are off by default; set PAPERQA_RUN_LLM_TESTS=1 to enable"


def chunk(index: int, text: str, title: str = "Attention Is All You Need") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        title=title,
        chunk_index=index,
        text=text,
        score=0.5,
    )


def service_with(
    chunks: list[RetrievedChunk] | None = None, answer: str = "A canned answer [1]."
) -> tuple[AnswerService, FakeRetrievalService, FakeLLMClient]:
    retrieval = FakeRetrievalService(chunks)
    llm = FakeLLMClient(answer)
    return AnswerService(retrieval, llm), retrieval, llm


# --- prompt construction ------------------------------------------------------


def test_prompt_contains_the_question() -> None:
    prompt = build_prompt("What is self-attention?", [chunk(0, "Self-attention relates.")])

    assert "What is self-attention?" in prompt


def test_prompt_contains_every_source_text() -> None:
    prompt = build_prompt(
        "q", [chunk(0, "first passage"), chunk(1, "second passage"), chunk(2, "third passage")]
    )

    assert "first passage" in prompt
    assert "second passage" in prompt
    assert "third passage" in prompt


def test_prompt_contains_the_document_title() -> None:
    """The title is what a reader uses to tell which paper a citation points at."""
    prompt = build_prompt("q", [chunk(0, "text", title="Dense Passage Retrieval")])

    assert "Dense Passage Retrieval" in prompt


def test_prompt_numbers_sources_from_one_in_retrieval_order() -> None:
    """[1] must label the best-scoring chunk: the numbers are what the answer cites back."""
    prompt = build_prompt(
        "q",
        [
            chunk(0, "best", title="First Paper"),
            chunk(1, "second", title="Second Paper"),
            chunk(2, "third", title="Third Paper"),
        ],
    )

    assert (
        prompt.index("[1] First Paper")
        < prompt.index("[2] Second Paper")
        < prompt.index("[3] Third Paper")
    )


def test_prompt_instructs_the_model_to_admit_missing_information() -> None:
    """Without this instruction the model answers from its own memory when the sources are
    irrelevant, and nothing downstream can tell that answer apart from a grounded one."""
    prompt = build_prompt("q", [chunk(0, "text")])

    assert "do not contain enough information" in prompt


def test_format_sources_of_nothing_is_an_empty_string() -> None:
    assert format_sources([]) == ""


# --- the service, against fakes ----------------------------------------------


def test_returns_the_answer_and_the_sources() -> None:
    chunks = [chunk(0, "first"), chunk(1, "second")]
    service, _, _ = service_with(chunks, answer="Because of [1].")

    result = service.answer("why?")

    assert result == AnswerResult(answer="Because of [1].", sources=chunks)


def test_passes_the_built_prompt_to_the_model() -> None:
    """Asserts on the exact prompt rather than on fragments: the pieces can each be
    present while the assembled prompt is missing a label or out of order."""
    chunks = [chunk(0, "first"), chunk(1, "second")]
    service, _, llm = service_with(chunks)

    service.answer("what is self-attention?")

    assert llm.prompts == [build_prompt("what is self-attention?", chunks)]


def test_citation_markers_resolve_against_the_returned_sources() -> None:
    """Pins the contract between the two halves of AnswerResult: [n] is sources[n - 1]."""
    chunks = [chunk(0, "alpha"), chunk(1, "beta"), chunk(2, "gamma")]
    service, _, llm = service_with(chunks)

    result = service.answer("q")
    prompt = llm.prompts[0]

    for position, source in enumerate(result.sources, start=1):
        assert source is chunks[position - 1]
        assert f"[{position}] {source.title}" in prompt
        assert source.text in prompt


def test_retrieves_once_with_the_requested_top_k() -> None:
    service, retrieval, _ = service_with([chunk(0, "text")])

    service.answer("q", top_k=7)

    assert retrieval.calls == [("q", 7)]


def test_uses_the_default_top_k_when_the_caller_does_not_specify() -> None:
    service, retrieval, _ = service_with([chunk(0, "text")])

    service.answer("q")

    assert retrieval.calls == [("q", DEFAULT_TOP_K)]


def test_asks_the_model_exactly_once() -> None:
    """Once per question: a second call would double the cost of every answer."""
    service, _, llm = service_with([chunk(0, "text")])

    service.answer("q")

    assert len(llm.prompts) == 1


def test_strips_surrounding_whitespace_from_the_answer() -> None:
    service, _, _ = service_with([chunk(0, "text")], answer="\n\n  Self-attention. \n")

    assert service.answer("q").answer == "Self-attention."


def test_returns_no_sources_when_retrieval_is_empty() -> None:
    service, _, _ = service_with([])

    assert service.answer("q").sources == []


def test_does_not_ask_the_model_when_there_are_no_sources() -> None:
    """With nothing to condition on the answer is already known, so the request would only
    add latency and an opportunity to answer from the model's own memory."""
    service, _, llm = service_with([])

    result = service.answer("q")

    assert llm.prompts == []
    assert result.answer == NO_SOURCES_ANSWER


# --- the real retrieval path: validation must precede generation --------------
#
# These use the real RetrievalService rather than the fake, because the property under
# test is that its validation runs before the model is called. A fake retrieval service
# that never raises would make that property untestable while looking covered.


def test_empty_question_raises_before_the_model_is_called(db_session: Session) -> None:
    llm = FakeLLMClient()
    service = AnswerService(
        RetrievalService(db_session, FakeEmbeddingClient()), llm
    )

    with pytest.raises(ValueError, match="question must not be empty"):
        service.answer("")

    assert llm.prompts == []


def test_whitespace_only_question_raises_before_the_model_is_called(
    db_session: Session,
) -> None:
    llm = FakeLLMClient()
    service = AnswerService(
        RetrievalService(db_session, FakeEmbeddingClient()), llm
    )

    with pytest.raises(ValueError, match="question must not be empty"):
        service.answer("   \n\t ")

    assert llm.prompts == []


def test_out_of_range_top_k_raises_before_the_model_is_called(db_session: Session) -> None:
    llm = FakeLLMClient()
    service = AnswerService(
        RetrievalService(db_session, FakeEmbeddingClient()), llm
    )

    with pytest.raises(ValueError, match="top_k must be between"):
        service.answer("q", top_k=MAX_TOP_K + 1)

    assert llm.prompts == []


def test_answers_from_chunks_retrieved_out_of_the_real_database(db_session: Session) -> None:
    """End to end through the real retrieval service, with only the model faked.

    Retrieval is exercised for real here so the two halves are proven to fit: the chunks
    that come back from ``search_similar`` are the ones the prompt is built from.
    """
    document = Document(title="A Real Paper", source="test", content_hash="a" * 64)
    db_session.add(document)
    db_session.flush()
    embedder = FakeEmbeddingClient()
    db_session.add(
        Chunk(
            document_id=document.id,
            chunk_index=0,
            text="self attention mechanism",
            token_count=3,
            embedding=embedder.embed_text("self attention mechanism"),
        )
    )
    db_session.flush()

    llm = FakeLLMClient(answer="It is [1].")
    service = AnswerService(RetrievalService(db_session, embedder), llm)

    result = service.answer("self attention mechanism", top_k=1)

    assert len(result.sources) == 1
    assert result.sources[0].title == "A Real Paper"
    assert "self attention mechanism" in llm.prompts[0]
    assert result.answer == "It is [1]."


# --- the LLM client, against a mock transport ---------------------------------
#
# MockTransport runs the real httpx request-building path without a network, so these
# cover the URL, headers and body rather than a hand-written stand-in for them.


def make_client(
    handler, api_key: str = "test-key", base_url: str = "https://example.invalid"
) -> tuple[LLMClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = LLMClient(
        api_key=api_key,
        model="test-model",
        base_url=base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(record)),
    )
    return client, requests


def ok_response(content: str = "The answer [1].") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_posts_to_the_chat_completions_endpoint() -> None:
    client, requests = make_client(lambda request: ok_response())

    client.generate("hello")

    assert str(requests[0].url) == "https://example.invalid/chat/completions"
    assert requests[0].method == "POST"


def test_trailing_slash_on_the_base_url_does_not_double_up() -> None:
    client, requests = make_client(
        lambda request: ok_response(), base_url="https://example.invalid/"
    )

    client.generate("hello")

    assert str(requests[0].url) == "https://example.invalid/chat/completions"


def test_sends_the_api_key_as_a_bearer_token() -> None:
    client, requests = make_client(lambda request: ok_response(), api_key="secret-value")

    client.generate("hello")

    assert requests[0].headers["authorization"] == "Bearer secret-value"


def test_sends_the_configured_model_and_the_prompt() -> None:
    client, requests = make_client(lambda request: ok_response())

    client.generate("What is self-attention?")

    body = json.loads(requests[0].content)
    assert body["model"] == "test-model"
    assert body["messages"] == [{"role": "user", "content": "What is self-attention?"}]


def test_bounds_the_answer_length() -> None:
    """Without a ceiling the cost of one answer is whatever the model decides to write."""
    client, requests = make_client(lambda request: ok_response())

    client.generate("hello")

    assert json.loads(requests[0].content)["max_tokens"] == MAX_ANSWER_TOKENS


def test_returns_the_message_content() -> None:
    client, _ = make_client(lambda request: ok_response("Self-attention relates positions."))

    assert client.generate("hello") == "Self-attention relates positions."


def test_raises_without_an_api_key_and_makes_no_request() -> None:
    client, requests = make_client(lambda request: ok_response(), api_key="")

    with pytest.raises(ValueError, match="llm_api_key"):
        client.generate("hello")

    assert requests == []


def test_raises_on_an_empty_prompt_and_makes_no_request() -> None:
    client, requests = make_client(lambda request: ok_response())

    with pytest.raises(ValueError, match="prompt must not be empty"):
        client.generate("   ")

    assert requests == []


def test_propagates_an_http_error() -> None:
    """A 401 or a rate limit must surface, not be read as an empty answer."""
    client, _ = make_client(
        lambda request: httpx.Response(401, json={"error": {"message": "bad key"}})
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.generate("hello")


# --- the process-wide client --------------------------------------------------


def test_get_llm_client_is_a_process_wide_singleton(monkeypatch) -> None:
    """One client per process. FastAPI's Depends calls this factory on every request, so a
    client built per call would open a new connection pool each time."""
    monkeypatch.setattr(llm, "_default_client", None)

    assert llm.get_llm_client() is llm.get_llm_client()


def test_get_llm_client_reads_the_configured_settings(monkeypatch) -> None:
    """A singleton built from the wrong settings would be invisible until the first request
    went to the wrong endpoint."""
    monkeypatch.setattr(llm, "_default_client", None)
    settings = get_settings()

    client = llm.get_llm_client()

    assert client.model == settings.llm_model
    assert client.base_url == settings.llm_base_url.rstrip("/")


# --- real LLM only: the prompt actually works ---------------------------------


@pytest.mark.skipif(not RUNS_REAL_LLM, reason=SKIP_REASON)
def test_real_model_answers_from_the_supplied_sources() -> None:
    """The claim the prompt makes is that answers come from the sources. This checks it
    against the real model, on a fact only the supplied source contains."""
    settings = get_settings()
    client = LLMClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    result = AnswerService(
        FakeRetrievalService(
            [chunk(0, "The Zorblax protocol uses eighteen parallel queues.", title="Invented")]
        ),
        client,
    ).answer("How many parallel queues does the Zorblax protocol use?")

    assert "eighteen" in result.answer.lower()
