import json
import wave
from pathlib import Path

import ffmpeg
import pytest

from glam import media
from glam.errors import GlamError
from glam.steps import tts


def make_tone_wav(path, duration, sample_rate=22050):
    (
        ffmpeg
        .input(f"sine=frequency=440:duration={duration}", f="lavfi")
        .output(str(path), format="wav", ar=sample_rate, ac=1, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )


class FakeSpeechResponse:
    def __init__(self, duration):
        self.duration = duration

    def stream_to_file(self, path):
        make_tone_wav(Path(path), self.duration)


class FakeSpeech:
    def __init__(self, durations):
        self._durations = list(durations)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        duration = self._durations.pop(0)
        return FakeSpeechResponse(duration)


class FakeClient:
    def __init__(self, durations):
        self.audio = _Audio(durations)


class _Audio:
    def __init__(self, durations):
        self.speech = FakeSpeech(durations)


def make_config(model="fake-tts", voice="default", **overrides):
    tts_cfg = {
        "backend": "openai_compatible",
        "base_url": "http://fake-host:9000/v1",
        "model": model,
        "voice": voice,
    }
    tts_cfg.update(overrides)
    return {"steps": {"tts": tts_cfg}}


def write_translation(job_dir, lang="ru", model="fake-model", segments=None):
    job_dir.mkdir(parents=True, exist_ok=True)
    segments = segments or [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "привет"},
        {"id": 1, "start": 1.0, "end": 2.0, "text": "мир"},
    ]
    path = job_dir / f"translation.{lang}.{model}.json"
    path.write_text(json.dumps({"model": model, "lang": lang, "segments": segments}))
    return path


def test_tts_writes_segment_clips_and_track(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir)

    fake_client = FakeClient([1.0, 1.0])
    monkeypatch.setattr(tts, "build_openai_client", lambda cfg: fake_client)

    track_path = tts.run(
        "myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    assert track_path == job_dir / "tts_track.ru.fake-tts.default.wav"
    assert track_path.exists()

    segments_dir = job_dir / "tts_segments.ru.fake-tts.default"
    assert (segments_dir / "0000.wav").exists()
    assert (segments_dir / "0001.wav").exists()

    track_duration = media.probe_duration(track_path)
    assert track_duration == pytest.approx(2.0, abs=0.1)

    assert len(fake_client.audio.speech.calls) == 2
    assert fake_client.audio.speech.calls[0]["model"] == "fake-tts"
    assert fake_client.audio.speech.calls[0]["voice"] == "default"
    assert fake_client.audio.speech.calls[0]["input"] == "привет"


def test_tts_empty_segment_text_writes_silence_without_calling_backend(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir, segments=[
        {"id": 0, "start": 0.0, "end": 1.0, "text": "  "},
        {"id": 1, "start": 1.0, "end": 2.0, "text": "мир"},
    ])

    fake_client = FakeClient([1.0])
    monkeypatch.setattr(tts, "build_openai_client", lambda cfg: fake_client)

    tts.run("myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)

    assert len(fake_client.audio.speech.calls) == 1  # only the non-empty segment


def test_tts_is_idempotent_at_track_level(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir)
    track_path = job_dir / "tts_track.ru.fake-tts.default.wav"
    job_dir.mkdir(parents=True, exist_ok=True)
    track_path.write_bytes(b"already there")

    fake_client = FakeClient([])
    monkeypatch.setattr(tts, "build_openai_client", lambda cfg: fake_client)

    result_path = tts.run(
        "myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    assert result_path == track_path
    assert track_path.read_bytes() == b"already there"
    assert fake_client.audio.speech.calls == []


def test_tts_reuses_existing_segment_clips_without_track(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir)
    segments_dir = job_dir / "tts_segments.ru.fake-tts.default"
    segments_dir.mkdir(parents=True)
    make_tone_wav(segments_dir / "0000.wav", 1.0)

    fake_client = FakeClient([1.0])  # only segment 1 needs synthesis
    monkeypatch.setattr(tts, "build_openai_client", lambda cfg: fake_client)

    tts.run("myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)

    assert len(fake_client.audio.speech.calls) == 1


def test_tts_force_regenerates_all_clips(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir)
    track_path = job_dir / "tts_track.ru.fake-tts.default.wav"
    job_dir.mkdir(parents=True, exist_ok=True)
    track_path.write_bytes(b"stale")

    fake_client = FakeClient([1.0, 1.0])
    monkeypatch.setattr(tts, "build_openai_client", lambda cfg: fake_client)

    tts.run(
        "myvid", make_config(), "ru", jobs_root=tmp_path / "jobs",
        force=True, echo=lambda *_: None,
    )

    assert len(fake_client.audio.speech.calls) == 2


def test_tts_missing_translation_raises(tmp_path, monkeypatch):
    (tmp_path / "jobs" / "myvid").mkdir(parents=True)
    fake_client = FakeClient([])
    monkeypatch.setattr(tts, "build_openai_client", lambda cfg: fake_client)

    with pytest.raises(GlamError):
        tts.run("myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)


def test_finalize_clip_corrects_duration_within_atempo_bound(tmp_path):
    raw = tmp_path / "raw.wav"
    out = tmp_path / "out.wav"
    make_tone_wav(raw, 1.1)  # 10% over target, well within default 1.3x bound

    tts._finalize_clip(raw, out, target_duration=1.0, sample_rate=22050, max_atempo=1.3)

    assert media.probe_duration(out) == pytest.approx(1.0, abs=0.05)


def test_finalize_clip_clamps_extreme_ratio_instead_of_forcing_exact_target(tmp_path):
    raw = tmp_path / "raw.wav"
    out = tmp_path / "out.wav"
    make_tone_wav(raw, 2.0)  # 2x target — beyond the 1.3x correction bound

    tts._finalize_clip(raw, out, target_duration=1.0, sample_rate=22050, max_atempo=1.3)

    result_duration = media.probe_duration(out)
    assert result_duration == pytest.approx(2.0 / 1.3, abs=0.05)
    assert result_duration > 1.0  # confirms it did NOT hit the target, by design


def test_assemble_track_inserts_silence_gaps(tmp_path):
    clip_a = tmp_path / "a.wav"
    clip_b = tmp_path / "b.wav"
    make_tone_wav(clip_a, 1.0, sample_rate=22050)
    make_tone_wav(clip_b, 1.0, sample_rate=22050)
    out = tmp_path / "track.wav"

    tts._assemble_track([clip_a, clip_b], [0.0, 3.0], out, sample_rate=22050)

    with wave.open(str(out), "rb") as f:
        total_duration = f.getnframes() / f.getframerate()
    assert total_duration == pytest.approx(4.0, abs=0.01)
