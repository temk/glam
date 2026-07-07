import openai
from dataclasses import dataclass

from glam.common.config import ServiceConfig
from glam.backend.params import parse_params
from glam.backend.tts.base import TtsBackend, TtsBackendError

# Local TTS backends can be slow; keep the same generous ceiling as `transcribe`.
REQUEST_TIMEOUT_SECONDS = 25 * 60


@dataclass
class OpenAITtsParams:
    model: str
    api_key: str | None = None
    voice: str | None = None


class OpenAITtsBackend(TtsBackend):
    def __init__(self, url: str, params: OpenAITtsParams) -> None:
        self._params = params
        # The OpenAI SDK requires a non-empty key even for keyless local backends.
        self._client = openai.OpenAI(
            base_url=url, api_key=params.api_key or "not-needed", timeout=REQUEST_TIMEOUT_SECONDS
        )

    @classmethod
    def from_service(cls, service: ServiceConfig) -> "OpenAITtsBackend":
        return cls(service.url, parse_params(service.params, OpenAITtsParams, TtsBackendError, "tts openai"))

    def synthesize(self, text: str, *, target: str, voice: str | None) -> bytes:
        # The OpenAI speech protocol has no language field; `target` is unused here.
        voice = voice or self._params.voice
        if not voice:
            raise TtsBackendError(
                "the openai tts protocol requires a voice: pass --voice, set the job voice, or set params.voice"
            )
        try:
            response = self._client.audio.speech.create(
                model=self._params.model, voice=voice, input=text, response_format="wav"
            )
        except openai.OpenAIError as e:
            raise TtsBackendError(f"tts service request failed: {e}") from e
        audio = getattr(response, "content", None)
        if not audio:
            raise TtsBackendError("tts service returned an empty response")
        return audio
