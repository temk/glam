import json
import openai
import pytest
from types import SimpleNamespace
from pathlib import Path

from glam.steps import transcribe as transcribe_step
from glam.common.job import JobInfo, Languages, SourceInfo, JobManifest, write_job_manifest
from glam.common.config import Config, Protocol, ConfigError, ServiceName, ServiceConfig
from glam.steps.transcribe import TranscribeError
from glam.backend.transcribe.base import TranscribeBackendError

DEFAULT_SEGMENTS = [
    {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello."},
    {"id": 1, "start": 1.5, "end": 3.0, "text": "World."},
]


def _make_job(tmp_path: Path, job_id: str = "jobA", source: str = "en", with_audio: bool = True) -> Path:
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
        languages=Languages(source=source, target="ru"),
    )
    write_job_manifest(manifest, job_path / "job.yaml")
    if with_audio:
        (job_path / "audio.wav").write_bytes(b"fake-wav")
    return job_path


def _config(tmp_path: Path, with_service: bool = True, model: str = "whisper-x") -> Config:
    services = []
    if with_service:
        services.append(
            ServiceConfig(
                name=ServiceName.TRANSCRIBE, protocol=Protocol.OPENAI, url="http://asr/v1", params={"model": model}
            )
        )
    return Config(services=services, job_dir=tmp_path)


def _fake_client(calls, segments=DEFAULT_SEGMENTS, error=None):
    response = SimpleNamespace(segments=None if segments is None else [SimpleNamespace(**s) for s in segments])

    def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return response

    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


@pytest.fixture
def patch_client(monkeypatch):
    """Replace the ASR client factory; return a recorder for the requests it receives."""

    def install(segments=DEFAULT_SEGMENTS, error=None):
        calls: list[dict] = []
        client = _fake_client(calls, segments=segments, error=error)
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
        return calls

    return install


def test_creates_transcript(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path)
    path = transcribe_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    data = json.loads(path.read_text())
    assert path.name == "transcript.json"
    assert data["version"] == 1
    assert data["step"] == "transcribe"
    assert data["job_id"] == "jobA"
    assert data["source_language"] == "en"
    assert data["model"] == "whisper-x"
    assert data["audio_artifact"] == "audio.wav"
    assert data["segments"] == DEFAULT_SEGMENTS


def test_requests_verbose_segments_in_source_language(tmp_path, patch_client):
    calls = patch_client()
    _make_job(tmp_path, source="de")
    transcribe_step.run("jobA", _config(tmp_path, model="whisper-x"), echo=lambda *_: None)

    assert len(calls) == 1
    req = calls[0]
    assert req["model"] == "whisper-x"
    assert req["language"] == "de"
    assert req["response_format"] == "verbose_json"
    assert req["timestamp_granularities"] == ["segment"]


def test_skips_when_transcript_exists(tmp_path, patch_client):
    calls = patch_client()
    job_path = _make_job(tmp_path)
    (job_path / "transcript.json").write_text('{"existing": true}')

    transcribe_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert calls == []  # ASR was not called
    assert json.loads((job_path / "transcript.json").read_text()) == {"existing": True}


def test_force_recreates(tmp_path, patch_client):
    calls = patch_client()
    job_path = _make_job(tmp_path)
    (job_path / "transcript.json").write_text('{"existing": true}')

    transcribe_step.run("jobA", _config(tmp_path), force=True, echo=lambda *_: None)

    assert len(calls) == 1
    assert json.loads((job_path / "transcript.json").read_text())["step"] == "transcribe"


def test_job_not_found(tmp_path, patch_client):
    patch_client()
    with pytest.raises(TranscribeError):
        transcribe_step.run("missing", _config(tmp_path), echo=lambda *_: None)


def test_missing_audio(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path, with_audio=False)
    with pytest.raises(TranscribeError):
        transcribe_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_missing_service_in_config(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path)
    with pytest.raises(ConfigError):
        transcribe_step.run("jobA", _config(tmp_path, with_service=False), echo=lambda *_: None)


@pytest.mark.parametrize("segments", [[], None])
def test_response_without_segments_raises(tmp_path, patch_client, segments):
    patch_client(segments=segments)
    _make_job(tmp_path)
    with pytest.raises(TranscribeBackendError):
        transcribe_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_service_unavailable_raises(tmp_path, patch_client):
    patch_client(error=openai.OpenAIError("connection refused"))
    _make_job(tmp_path)
    with pytest.raises(TranscribeBackendError):
        transcribe_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
