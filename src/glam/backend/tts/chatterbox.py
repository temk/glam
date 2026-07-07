import httpx

from glam.common.config import ServiceConfig
from glam.backend.tts.base import TtsBackend, TtsBackendError

# Local TTS backends can be slow; keep the same generous ceiling as `transcribe`.
REQUEST_TIMEOUT_SECONDS = 25 * 60

# The native `/tts` requires a `predefined_voice_id` (the server has no implicit default), so when
# no voice is resolved we fall back to this built-in predefined voice from the server's voice set.
DEFAULT_VOICE = "Michael.wav"


class ChatterboxTtsBackend(TtsBackend):
    """Native Chatterbox-TTS-Server backend (`POST /tts`), giving per-request language and cloning."""

    def __init__(self, url: str) -> None:
        self._endpoint = url.rstrip("/") + "/tts"
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)

    @classmethod
    def from_service(cls, service: ServiceConfig) -> "ChatterboxTtsBackend":
        return cls(service.url)

    def synthesize(self, text: str, *, target: str, voice: str | None) -> bytes:
        # Send the target as `language` for multilingual dubbing. Generative params are left at the
        # server's defaults; a predefined voice is always sent (required by the server).
        payload: dict = {
            "text": text,
            "language": target,
            "voice_mode": "predefined",
            "predefined_voice_id": voice or DEFAULT_VOICE,
            "output_format": "wav",
            "stream": False,
        }
        try:
            response = self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise TtsBackendError(f"tts service request failed: {e}") from e
        if not response.content:
            raise TtsBackendError("tts service returned an empty response")
        return response.content
