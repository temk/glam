import io
import json
import wave
import httpx
import openai
import pytest
from types import SimpleNamespace
from pathlib import Path

from glam.steps import tts as tts_step
from glam.steps.tts import TtsError
from glam.common.job import JobInfo, Languages, SourceInfo, JobManifest, write_job_manifest
from glam.common.config import Config, Protocol, ConfigError, ServiceName, ServiceConfig
from glam.backend.tts.base import TtsBackendError
from glam.common.translation import TranslationError
from glam.backend.tts.chatterbox import DEFAULT_VOICE as CHATTERBOX_DEFAULT_VOICE

FRAMERATE = 16000

TRANSLATION = {
    "version": 1,
    "step": "translate",
    "job_id": "jobA",
    "source_language": "en",
    "model": "llm-x",
    "audio_artifact": "audio.wav",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello.", "translated_text": "Привет."},
        {"id": 1, "start": 1.5, "end": 3.0, "text": "World.", "translated_text": "Мир."},
    ],
}


def _wav_bytes(seconds: float, framerate: int = FRAMERATE, nchannels: int = 1, sampwidth: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        # Non-zero samples so synthesized audio is distinguishable from inserted silence.
        w.writeframes(b"\x11\x11" * (round(seconds * framerate) * nchannels))
    return buf.getvalue()


def _wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()


def _make_job(
    tmp_path: Path, job_id: str = "jobA", translation: dict | None = TRANSLATION, voice: str | None = None
) -> Path:
    job_path = tmp_path / job_id
    job_path.mkdir(parents=True)
    manifest = JobManifest(
        version=1,
        job=JobInfo(id=job_id, created_at="2026-07-05T00:00:00+00:00"),
        source=SourceInfo(
            original_path="/x/video.mp4",
            filename="video.mp4",
            artifact="source.mp4",
            audio_artifact="audio.wav",
            duration_seconds=3.0,
        ),
        languages=Languages(source="en", target="ru"),
        voice=voice,
    )
    write_job_manifest(manifest, job_path / "job.yaml")
    if translation is not None:
        (job_path / "translation.ru.json").write_text(json.dumps(translation, ensure_ascii=False, indent=2))
    return job_path


def _chatterbox_config(tmp_path: Path, with_service: bool = True) -> Config:
    services = []
    if with_service:
        services.append(ServiceConfig(name=ServiceName.TTS, protocol=Protocol.CHATTERBOX, url="http://tts:8004"))
    return Config(services=services, job_dir=tmp_path)


def _openai_config(tmp_path: Path, params: dict | None = None) -> Config:
    service = ServiceConfig(
        name=ServiceName.TTS, protocol=Protocol.OPENAI, url="http://tts/v1", params=params or {"model": "tts-x"}
    )
    return Config(services=[service], job_dir=tmp_path)


# --- fake backends ---


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


@pytest.fixture
def patch_chatterbox(monkeypatch):
    """Patch httpx.Client used by the chatterbox backend; return recorded POST payloads."""

    def install(audio=None, post_error=None):
        calls: list[dict] = []

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def post(self, url, json):
                calls.append({"url": url, "json": json})
                if post_error is not None:
                    raise post_error
                content = audio(json) if callable(audio) else _wav_bytes(0.5)
                return _FakeResponse(content)

        monkeypatch.setattr("httpx.Client", FakeClient)
        return calls

    return install


@pytest.fixture
def patch_openai(monkeypatch):
    """Patch openai.OpenAI used by the openai backend; return recorded request kwargs."""

    def install(audio=None, error=None):
        calls: list[dict] = []

        def create(**kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            content = audio(kwargs) if callable(audio) else _wav_bytes(0.5)
            return SimpleNamespace(content=content)

        client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
        return calls

    return install


# --- protocol-agnostic step behavior (exercised through the chatterbox backend) ---


def test_creates_audio_track(tmp_path, patch_chatterbox):
    patch_chatterbox()
    _make_job(tmp_path)
    path = tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    assert path.name == "tts.ru.wav"
    assert path.exists()


def test_uses_translated_text(tmp_path, patch_chatterbox):
    calls = patch_chatterbox()
    _make_job(tmp_path)
    tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    assert [c["json"]["text"] for c in calls] == ["Привет.", "Мир."]


def test_prefers_fixed_translation_when_present(tmp_path, patch_chatterbox):
    calls = patch_chatterbox()
    job_path = _make_job(tmp_path)  # writes translation.ru.json
    fixed = {
        **TRANSLATION,
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.5, "translated_text": "Приве́т."},
            {"id": 1, "start": 1.5, "end": 3.0, "translated_text": "Ми́р."},
        ],
    }
    (job_path / "translation.ru.fixed.json").write_text(json.dumps(fixed, ensure_ascii=False))
    tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    assert [c["json"]["text"] for c in calls] == ["Приве́т.", "Ми́р."]  # from the .fixed.json


def test_preserves_segment_order(tmp_path, patch_chatterbox):
    calls = patch_chatterbox()
    translation = {
        **TRANSLATION,
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "translated_text": "первый"},
            {"id": 1, "start": 1.0, "end": 2.0, "translated_text": "второй"},
            {"id": 2, "start": 2.0, "end": 3.0, "translated_text": "третий"},
        ],
    }
    _make_job(tmp_path, translation=translation)
    tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    assert [c["json"]["text"] for c in calls] == ["первый", "второй", "третий"]


def test_inserts_silence_between_segments(tmp_path, patch_chatterbox):
    translation = {
        **TRANSLATION,
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "translated_text": "a"},
            {"id": 1, "start": 2.0, "end": 3.0, "translated_text": "b"},
        ],
    }
    patch_chatterbox(audio=lambda payload: _wav_bytes(0.5))
    job_path = _make_job(tmp_path, translation=translation)
    path = tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    # 0.5 (frag) + 1.5 (silence to reach start=2.0) + 0.5 (frag) = 2.5s
    assert _wav_duration((job_path / path.name).read_bytes()) == pytest.approx(2.5, abs=1e-3)


def test_overlong_fragment_pushes_later_without_trimming(tmp_path, patch_chatterbox):
    translation = {
        **TRANSLATION,
        "segments": [
            {"id": 0, "start": 0.0, "end": 0.4, "translated_text": "long"},
            {"id": 1, "start": 0.4, "end": 1.0, "translated_text": "next"},
        ],
    }
    durations = iter([1.0, 0.5])
    patch_chatterbox(audio=lambda payload: _wav_bytes(next(durations)))
    job_path = _make_job(tmp_path, translation=translation)
    path = tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    # No silence, no trimming: total is exactly the two fragments back to back (1.0 + 0.5).
    assert _wav_duration((job_path / path.name).read_bytes()) == pytest.approx(1.5, abs=1e-3)


def test_format_mismatch_is_assembly_error(tmp_path, patch_chatterbox):
    rates = iter([16000, 8000])
    patch_chatterbox(audio=lambda payload: _wav_bytes(0.5, framerate=next(rates)))
    _make_job(tmp_path)
    with pytest.raises(TtsError, match="unable to assemble"):
        tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)


def test_skips_if_exists(tmp_path, patch_chatterbox):
    calls = patch_chatterbox()
    job_path = _make_job(tmp_path)
    (job_path / "tts.ru.wav").write_bytes(b"stale")

    tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)
    assert (job_path / "tts.ru.wav").read_bytes() == b"stale"
    assert calls == []  # no synthesis when skipping


def test_force_recreates(tmp_path, patch_chatterbox):
    patch_chatterbox()
    job_path = _make_job(tmp_path)
    (job_path / "tts.ru.wav").write_bytes(b"stale")

    tts_step.run("jobA", _chatterbox_config(tmp_path), force=True, echo=lambda *_: None)
    assert (job_path / "tts.ru.wav").read_bytes() != b"stale"


def test_target_override_reads_and_names_that_language(tmp_path, patch_chatterbox):
    calls = patch_chatterbox()
    job_path = _make_job(tmp_path)  # translation.ru.json; job target ru
    (job_path / "translation.de.json").write_text(
        json.dumps({"segments": [{"id": 0, "start": 0.0, "end": 1.0, "translated_text": "Hallo"}]})
    )
    path = tts_step.run("jobA", _chatterbox_config(tmp_path), target="de", echo=lambda *_: None)

    assert path.name == "tts.de.wav"
    assert calls[0]["json"]["language"] == "de"


def test_reports_progress(tmp_path, patch_chatterbox):
    patch_chatterbox()
    _make_job(tmp_path)
    messages: list[str] = []
    tts_step.run("jobA", _chatterbox_config(tmp_path), echo=messages.append)

    assert any("synthesizing segment 1/2" in m for m in messages)
    assert any("wrote" in m and "tts.ru.wav" in m for m in messages)


def test_missing_translation(tmp_path, patch_chatterbox):
    patch_chatterbox()
    job_path = _make_job(tmp_path)
    (job_path / "translation.ru.json").unlink()
    with pytest.raises(TranslationError, match="missing translation artifact"):
        tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)


def test_invalid_translation_format(tmp_path, patch_chatterbox):
    patch_chatterbox()
    job_path = _make_job(tmp_path)
    (job_path / "translation.ru.json").write_text("{ not json")
    with pytest.raises(TranslationError, match="invalid translation"):
        tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)


def test_missing_translated_text(tmp_path, patch_chatterbox):
    patch_chatterbox()
    translation = {**TRANSLATION, "segments": [{"id": 0, "start": 0.0, "end": 1.0}]}
    _make_job(tmp_path, translation=translation)
    with pytest.raises(TranslationError, match="missing 'translated_text'"):
        tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)


def test_invalid_timestamps(tmp_path, patch_chatterbox):
    patch_chatterbox()
    translation = {**TRANSLATION, "segments": [{"id": 0, "start": 2.0, "end": 1.0, "translated_text": "t"}]}
    _make_job(tmp_path, translation=translation)
    with pytest.raises(TranslationError, match="end <= start"):
        tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)


def test_missing_tts_service(tmp_path, patch_chatterbox):
    patch_chatterbox()
    _make_job(tmp_path)
    with pytest.raises(ConfigError, match="not defined in the config"):
        tts_step.run("jobA", _chatterbox_config(tmp_path, with_service=False), echo=lambda *_: None)


# --- chatterbox protocol specifics ---


def test_chatterbox_posts_to_tts_endpoint_with_language(tmp_path, patch_chatterbox):
    calls = patch_chatterbox()
    _make_job(tmp_path)
    tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    assert all(c["url"] == "http://tts:8004/tts" for c in calls)
    assert all(c["json"]["language"] == "ru" for c in calls)
    assert all(c["json"]["output_format"] == "wav" for c in calls)


def test_chatterbox_uses_default_voice_when_unset(tmp_path, patch_chatterbox):
    calls = patch_chatterbox()
    _make_job(tmp_path)  # no --voice, no job voice
    path = tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    # The server requires a predefined voice, so the backend falls back to its built-in default;
    # a fallback default does not enter the artifact name.
    assert path.name == "tts.ru.wav"
    assert all(c["json"]["predefined_voice_id"] == CHATTERBOX_DEFAULT_VOICE for c in calls)


def test_chatterbox_sends_voice_when_set(tmp_path, patch_chatterbox):
    calls = patch_chatterbox()
    _make_job(tmp_path, voice="female_01")  # voice from job.yaml
    path = tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)

    assert path.name == "tts.ru.female_01.wav"
    assert all(c["json"]["predefined_voice_id"] == "female_01" for c in calls)


def test_chatterbox_backend_unavailable(tmp_path, patch_chatterbox):
    patch_chatterbox(post_error=httpx.ConnectError("connection refused"))
    _make_job(tmp_path)
    with pytest.raises(TtsBackendError, match="request failed"):
        tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)


def test_chatterbox_invalid_response(tmp_path, patch_chatterbox):
    patch_chatterbox(audio=lambda payload: b"not a wav file")
    _make_job(tmp_path)
    with pytest.raises(TtsError, match="invalid response"):
        tts_step.run("jobA", _chatterbox_config(tmp_path), echo=lambda *_: None)


# --- openai protocol specifics ---


def test_openai_requires_voice(tmp_path, patch_openai):
    patch_openai()
    _make_job(tmp_path)  # no voice anywhere
    with pytest.raises(TtsBackendError, match="requires a voice"):
        tts_step.run("jobA", _openai_config(tmp_path), echo=lambda *_: None)


def test_openai_uses_params_voice(tmp_path, patch_openai):
    calls = patch_openai()
    _make_job(tmp_path)
    path = tts_step.run(
        "jobA", _openai_config(tmp_path, params={"model": "tts-x", "voice": "alloy"}), echo=lambda *_: None
    )

    # params.voice fills the request but is a config default, so it does not enter the artifact name.
    assert path.name == "tts.ru.wav"
    assert all(c["voice"] == "alloy" for c in calls)
    assert all(c["model"] == "tts-x" for c in calls)
    assert [c["input"] for c in calls] == ["Привет.", "Мир."]


def test_openai_cli_voice_overrides_params_voice(tmp_path, patch_openai):
    calls = patch_openai()
    _make_job(tmp_path, voice="echo")  # job voice
    path = tts_step.run(
        "jobA",
        _openai_config(tmp_path, params={"model": "tts-x", "voice": "alloy"}),
        voice="nova",
        echo=lambda *_: None,
    )

    assert path.name == "tts.ru.nova.wav"  # --voice wins and enters the name
    assert all(c["voice"] == "nova" for c in calls)


def test_openai_backend_unavailable(tmp_path, patch_openai):
    patch_openai(error=openai.OpenAIError("connection refused"))
    _make_job(tmp_path, voice="alloy")
    with pytest.raises(TtsBackendError, match="request failed"):
        tts_step.run("jobA", _openai_config(tmp_path), echo=lambda *_: None)
