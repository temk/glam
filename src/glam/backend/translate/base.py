from abc import ABC, abstractmethod
from dataclasses import dataclass

from glam.common.config import Protocol, ServiceConfig
from glam.common.errors import GlamError


class TranslateBackendError(GlamError):
    pass


@dataclass
class ChatResult:
    content: str | None
    finish_reason: str | None


class TranslateBackend(ABC):
    model: str  # model label recorded in dump files

    @abstractmethod
    def complete(self, messages: list[dict], response_format: dict) -> ChatResult:
        """Run one chat completion and return its content and finish reason."""


def build_translate_backend(service: ServiceConfig) -> TranslateBackend:
    # Import the selected implementation lazily so picking one backend never imports another's SDK.
    if service.protocol is Protocol.OPENAI:
        from glam.backend.translate.openai import OpenAITranslateBackend

        return OpenAITranslateBackend.from_service(service)
    raise TranslateBackendError(f"translate does not support protocol '{service.protocol.value}'")
