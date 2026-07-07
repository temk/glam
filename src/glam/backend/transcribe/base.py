from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

from glam.common.config import Protocol, ServiceConfig
from glam.common.errors import GlamError


class TranscribeBackendError(GlamError):
    pass


@dataclass
class AsrSegment:
    id: int
    start: float
    end: float
    text: str


class TranscribeBackend(ABC):
    model: str  # model label recorded in the transcript artifact

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str) -> list[AsrSegment]:
        """Transcribe audio into segments with timestamps in the source language."""


def build_transcribe_backend(service: ServiceConfig) -> TranscribeBackend:
    # Import the selected implementation lazily so picking one backend never imports another's SDK.
    if service.protocol is Protocol.OPENAI:
        from glam.backend.transcribe.openai import OpenAITranscribeBackend

        return OpenAITranscribeBackend.from_service(service)
    raise TranscribeBackendError(f"transcribe does not support protocol '{service.protocol.value}'")
