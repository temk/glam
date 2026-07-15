import wave
import openai
from pathlib import Path
from dataclasses import dataclass

from glam.common.config import ServiceConfig
from glam.backend.params import parse_params
from glam.backend.transcribe.base import AsrSegment, TranscribeBackend, TranscribeBackendError

# A local ASR backend runs at roughly realtime or faster, so the request timeout scales with the audio
# rather than using the SDK's ~10 min default. The floor covers short clips, where model load dominates.
MIN_REQUEST_TIMEOUT_SECONDS = 3 * 60


@dataclass
class OpenAITranscribeParams:
    model: str
    api_key: str | None = None


class OpenAITranscribeBackend(TranscribeBackend):
    def __init__(self, url: str, params: OpenAITranscribeParams) -> None:
        self.model = params.model
        # The OpenAI SDK requires a non-empty key even for keyless local backends.
        self._client = openai.OpenAI(base_url=url, api_key=params.api_key or "not-needed")

    @classmethod
    def from_service(cls, service: ServiceConfig) -> "OpenAITranscribeBackend":
        params = parse_params(service.params, OpenAITranscribeParams, TranscribeBackendError, "transcribe openai")
        return cls(service.url, params)

    def transcribe(self, audio_path: Path, language: str) -> list[AsrSegment]:
        timeout = max(MIN_REQUEST_TIMEOUT_SECONDS, _audio_duration_seconds(audio_path))
        try:
            with audio_path.open("rb") as audio_file:
                response = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    extra_body={"condition_on_previous_text": False, "vad_filter": True},
                    timeout=timeout,
                )
        except openai.OpenAIError as e:
            raise TranscribeBackendError(f"ASR service request failed: {e}") from e
        return _extract_segments(response)


def _audio_duration_seconds(audio_path: Path) -> float:
    """The pipeline's audio artifact is always WAV, so the duration comes from its header, not from ffprobe."""
    try:
        with wave.open(str(audio_path)) as w:
            framerate = w.getframerate()
            if framerate <= 0:
                raise TranscribeBackendError(f"audio has no frame rate: {audio_path}")
            return w.getnframes() / framerate
    except (wave.Error, OSError) as e:
        raise TranscribeBackendError(f"cannot read audio duration from {audio_path}: {e}") from e


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
