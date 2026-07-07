import openai
from dataclasses import dataclass

from glam.common.config import ServiceConfig
from glam.backend.params import parse_params
from glam.backend.translate.base import ChatResult, TranslateBackend, TranslateBackendError

# Local LLM backends can be slow; keep the same generous ceiling as `transcribe`.
REQUEST_TIMEOUT_SECONDS = 25 * 60


@dataclass
class OpenAITranslateParams:
    model: str
    api_key: str | None = None


class OpenAITranslateBackend(TranslateBackend):
    def __init__(self, url: str, params: OpenAITranslateParams) -> None:
        self.model = params.model
        # The OpenAI SDK requires a non-empty key even for keyless local backends.
        self._client = openai.OpenAI(
            base_url=url, api_key=params.api_key or "not-needed", timeout=REQUEST_TIMEOUT_SECONDS
        )

    @classmethod
    def from_service(cls, service: ServiceConfig) -> "OpenAITranslateBackend":
        params = parse_params(service.params, OpenAITranslateParams, TranslateBackendError, "translate openai")
        return cls(service.url, params)

    def complete(self, messages: list[dict], response_format: dict) -> ChatResult:
        try:
            # The SDK's typed overloads reject our plain dict `messages`/`response_format`, which are
            # built dynamically and validated server-side; the runtime call is correct.
            response = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model, temperature=0, response_format=response_format, messages=messages
            )
        except openai.OpenAIError as e:
            raise TranslateBackendError(f"translation service request failed: {e}") from e
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice else None
        finish_reason = getattr(choice, "finish_reason", None)
        return ChatResult(content=content, finish_reason=finish_reason)
