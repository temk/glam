from abc import ABC, abstractmethod

from glam.common.config import Protocol, ServiceConfig
from glam.common.errors import GlamError


class TtsBackendError(GlamError):
    pass


class TtsBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, *, target: str, voice: str | None) -> bytes:
        """Return WAV audio bytes for one segment's text."""


def build_tts_backend(service: ServiceConfig) -> TtsBackend:
    # Import the selected implementation lazily so picking one backend never imports another's SDK.
    if service.protocol is Protocol.OPENAI:
        from glam.backend.tts.openai import OpenAITtsBackend

        return OpenAITtsBackend.from_service(service)
    if service.protocol is Protocol.CHATTERBOX:
        from glam.backend.tts.chatterbox import ChatterboxTtsBackend

        return ChatterboxTtsBackend.from_service(service)
    raise TtsBackendError(f"tts does not support protocol '{service.protocol.value}'")
