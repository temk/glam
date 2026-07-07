import openai
from pathlib import Path
from dataclasses import dataclass

from glam.common.config import ServiceConfig
from glam.backend.params import parse_params
from glam.backend.transcribe.base import AsrSegment, TranscribeBackend, TranscribeBackendError

# Local ASR backends can take much longer than the SDK's ~10 min default on long audio.
REQUEST_TIMEOUT_SECONDS = 25 * 60


@dataclass
class OpenAITranscribeParams:
    model: str
    api_key: str | None = None


class OpenAITranscribeBackend(TranscribeBackend):
    def __init__(self, url: str, params: OpenAITranscribeParams) -> None:
        self.model = params.model
        # The OpenAI SDK requires a non-empty key even for keyless local backends.
        self._client = openai.OpenAI(
            base_url=url, api_key=params.api_key or "not-needed", timeout=REQUEST_TIMEOUT_SECONDS
        )

    @classmethod
    def from_service(cls, service: ServiceConfig) -> "OpenAITranscribeBackend":
        params = parse_params(service.params, OpenAITranscribeParams, TranscribeBackendError, "transcribe openai")
        return cls(service.url, params)

    def transcribe(self, audio_path: Path, language: str) -> list[AsrSegment]:
        try:
            with audio_path.open("rb") as audio_file:
                response = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    extra_body={"condition_on_previous_text": False, "vad_filter": True},
                )
        except openai.OpenAIError as e:
            raise TranscribeBackendError(f"ASR service request failed: {e}") from e
        return _extract_segments(response)


def _extract_segments(response) -> list[AsrSegment]:
    segments = getattr(response, "segments", None)
    if not segments:
        raise TranscribeBackendError("ASR response has no segment-level timestamps")
    try:
        return [
            AsrSegment(id=getattr(seg, "id", i), start=float(seg.start), end=float(seg.end), text=str(seg.text))
            for i, seg in enumerate(segments)
        ]
    except (AttributeError, TypeError, ValueError) as e:
        raise TranscribeBackendError(f"invalid segment in ASR response: {e}") from e
