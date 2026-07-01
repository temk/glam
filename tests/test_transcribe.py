import json
from unittest.mock import MagicMock

import pytest

from glam.steps import transcribe


class FakeTranscription:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def make_config(model="fake-model", **overrides):
    asr = {
        "backend": "openai_compatible",
        "base_url": "http://fake-host:8000/v1",
        "model": model,
    }
    asr.update(overrides)
    return {"steps": {"asr": asr}}


def make_job(tmp_path, video_id="myvid"):
    job_dir = tmp_path / "jobs" / video_id
    job_dir.mkdir(parents=True)
    (job_dir / "audio.wav").write_bytes(b"fake audio bytes")
    return job_dir


def test_transcribe_writes_segments(tmp_path, monkeypatch):
    job_dir = make_job(tmp_path)
    fake_response = FakeTranscription({
        "text": "hello world",
        "language": "en",
        "duration": 1.2,
        "segments": [{"id": 0, "start": 0.0, "end": 1.2, "text": "hello world"}],
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.2},
        ],
    })
    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response
    monkeypatch.setattr(transcribe, "build_openai_client", lambda cfg: fake_client)

    result_path = transcribe.run(
        "myvid", make_config(), jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    assert result_path == job_dir / "transcript.fake-model.json"
    data = json.loads(result_path.read_text())
    assert data["model"] == "fake-model"
    assert data["segments"][0]["text"] == "hello world"
    assert len(data["words"]) == 2

    _, kwargs = fake_client.audio.transcriptions.create.call_args
    assert kwargs["model"] == "fake-model"
    assert kwargs["response_format"] == "verbose_json"
    assert kwargs["timestamp_granularities"] == ["segment", "word"]


def test_transcribe_missing_audio_raises(tmp_path):
    with pytest.raises(transcribe.TranscribeError):
        transcribe.run(
            "missing", make_config(), jobs_root=tmp_path / "jobs", echo=lambda *_: None
        )


def test_transcribe_is_idempotent_unless_forced(tmp_path, monkeypatch):
    job_dir = make_job(tmp_path)
    transcript_path = job_dir / "transcript.fake-model.json"
    transcript_path.write_text('{"already": "there"}')

    fake_client = MagicMock()
    monkeypatch.setattr(transcribe, "build_openai_client", lambda cfg: fake_client)

    result_path = transcribe.run(
        "myvid", make_config(), jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    assert result_path == transcript_path
    fake_client.audio.transcriptions.create.assert_not_called()


def test_transcribe_force_recomputes(tmp_path, monkeypatch):
    job_dir = make_job(tmp_path)
    transcript_path = job_dir / "transcript.fake-model.json"
    transcript_path.write_text('{"already": "there"}')

    fake_response = FakeTranscription({"segments": []})
    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response
    monkeypatch.setattr(transcribe, "build_openai_client", lambda cfg: fake_client)

    transcribe.run(
        "myvid", make_config(), jobs_root=tmp_path / "jobs", force=True, echo=lambda *_: None
    )

    fake_client.audio.transcriptions.create.assert_called_once()
    assert json.loads(transcript_path.read_text())["segments"] == []
