"""Turn a prompt into an answer.

A plain HTTP call rather than a vendor SDK: the endpoint is OpenAI-compatible, httpx is
already a dependency, and a separate package would be one more thing to keep in step for
no gain at this size.
"""

import httpx

from app.core.config import get_settings

# Ceiling on the answer, in tokens. The prompt asks for a concise answer; this bounds what
# a model ignoring that instruction can cost, in both money and latency.
MAX_ANSWER_TOKENS = 1024

# Seconds to wait for a completion. Generation dominates the request, so this is generous:
# long enough for a full answer, short enough that a hung connection does not hold a
# worker indefinitely.
REQUEST_TIMEOUT_SECONDS = 60.0


class LLMClient:
    """A thin wrapper around an OpenAI-compatible chat completions endpoint.

    ``http_client`` is injected so tests can pass one backed by a mock transport and
    exercise this class's real request-building code without a network call.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        # Stored without a trailing slash so the path is joined in exactly one place.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = http_client or httpx.Client()

    def generate(self, prompt: str) -> str:
        """Return the model's answer to ``prompt``.

        A missing key or an empty prompt raises here rather than being sent: both would
        come back as a 401 or as an answer to nothing, costing a round trip and pointing
        at the wrong place.
        """
        if not self.api_key:
            raise ValueError("llm_api_key must be set (see .env.example)")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_ANSWER_TOKENS,
            },
            timeout=self.timeout,
        )
        # HTTP errors propagate as httpx.HTTPStatusError, which carries the status and the
        # provider's message. Translating them here would discard the only diagnostic the
        # provider gives.
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# Process-wide instance, for the same reason as the embedding client: callers under a web
# request must share one, and the leading underscore marks it as internal state.
_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the process-wide LLMClient, creating it on first use.

    Zero-argument, so FastAPI's ``Depends`` accepts it and this module stays free of any
    web-framework import. Building it is cheap, but keeping it lazy means a process that
    never answers a question never reads the LLM configuration at all.
    """
    global _default_client
    if _default_client is None:
        settings = get_settings()
        _default_client = LLMClient(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )
    return _default_client
