import json
import pytest
from pathlib import Path

from glam.steps import subtitles as subtitles_step
from glam.common.job import JobInfo, Languages, SourceInfo, JobManifest, write_job_manifest
from glam.common.config import Config
from glam.steps.subtitles import SubtitlesError
from glam.common.translation import TranslationError

TRANSLATION = {
    "version": 1,
    "step": "translate",
    "job_id": "jobA",
    "source_language": "en",
    "model": "llm-x",
    "audio_artifact": "audio.wav",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello.", "translated_text": "Привет."},
        {"id": 1, "start": 1.5, "end": 3.25, "text": "World.", "translated_text": "Мир."},
    ],
}


def _make_job(tmp_path: Path, job_id: str = "jobA", translation: dict | None = TRANSLATION) -> Path:
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
            duration_seconds=3.25,
        ),
        languages=Languages(source="en", target="ru"),
    )
    write_job_manifest(manifest, job_path / "job.yaml")
    if translation is not None:
        (job_path / "translation.ru.json").write_text(json.dumps(translation, ensure_ascii=False, indent=2) + "\n")
    return job_path


def _config(tmp_path: Path) -> Config:
    return Config(services=[], job_dir=tmp_path)


def _blocks(text: str) -> list[str]:
    return text.strip("\n").split("\n\n")


def test_creates_subtitles(tmp_path):
    _make_job(tmp_path)
    path = subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert path.name == "subtitles.ru.srt"
    assert path.exists()


def test_target_override_reads_and_writes_that_language(tmp_path):
    job_path = _make_job(tmp_path)  # writes translation.ru.json; job target is ru
    # Provide a German translation artifact and render it via --target de.
    (job_path / "translation.de.json").write_text(
        json.dumps({"segments": [{"id": 0, "start": 0.0, "end": 1.0, "translated_text": "Hallo"}]})
    )
    path = subtitles_step.run("jobA", _config(tmp_path), target="de", echo=lambda *_: None)

    assert path.name == "subtitles.de.srt"
    assert "Hallo" in path.read_text()


def test_cue_numbering_and_timestamps(tmp_path):
    _make_job(tmp_path)
    path = subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    blocks = _blocks(path.read_text())
    assert len(blocks) == 2
    assert blocks[0].splitlines()[0] == "1"
    assert blocks[0].splitlines()[1] == "00:00:00,000 --> 00:00:01,500"
    assert blocks[1].splitlines()[0] == "2"
    assert blocks[1].splitlines()[1] == "00:00:01,500 --> 00:00:03,250"


def test_uses_translated_text(tmp_path):
    _make_job(tmp_path)
    path = subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    content = path.read_text()
    assert "Привет." in content
    assert "Мир." in content
    assert "Hello." not in content


def test_preserves_segment_order(tmp_path):
    reversed_segments = {
        **TRANSLATION,
        "segments": [
            {"id": 0, "start": 5.0, "end": 6.0, "text": "b", "translated_text": "второй"},
            {"id": 1, "start": 0.0, "end": 1.0, "text": "a", "translated_text": "первый"},
        ],
    }
    _make_job(tmp_path, translation=reversed_segments)
    path = subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    blocks = _blocks(path.read_text())
    assert "второй" in blocks[0]
    assert "первый" in blocks[1]


def test_splits_long_text_into_readable_lines(tmp_path):
    long_text = " ".join(["слово"] * 40)
    translation = {**TRANSLATION, "segments": [{"id": 0, "start": 0.0, "end": 2.0, "translated_text": long_text}]}
    _make_job(tmp_path, translation=translation)
    path = subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    text_lines = _blocks(path.read_text())[0].splitlines()[2:]
    assert len(text_lines) > 1
    assert all(len(line) <= subtitles_step.MAX_LINE_LENGTH for line in text_lines)


def test_hour_scale_timestamp(tmp_path):
    translation = {**TRANSLATION, "segments": [{"id": 0, "start": 3661.5, "end": 3663.0, "translated_text": "x"}]}
    _make_job(tmp_path, translation=translation)
    path = subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert "01:01:01,500 --> 01:01:03,000" in path.read_text()


def test_skips_if_exists(tmp_path):
    job_path = _make_job(tmp_path)
    subtitles_path = job_path / "subtitles.ru.srt"
    subtitles_path.write_text("stale")

    subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
    assert subtitles_path.read_text() == "stale"


def test_force_recreates(tmp_path):
    job_path = _make_job(tmp_path)
    subtitles_path = job_path / "subtitles.ru.srt"
    subtitles_path.write_text("stale")

    subtitles_step.run("jobA", _config(tmp_path), force=True, echo=lambda *_: None)
    assert subtitles_path.read_text() != "stale"
    assert "Привет." in subtitles_path.read_text()


def test_missing_job(tmp_path):
    with pytest.raises(SubtitlesError, match="job not found"):
        subtitles_step.run("nope", _config(tmp_path), echo=lambda *_: None)


def test_missing_translation(tmp_path):
    job_path = _make_job(tmp_path)
    (job_path / "translation.ru.json").unlink()
    with pytest.raises(TranslationError, match="missing translation artifact"):
        subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_invalid_translation_json(tmp_path):
    job_path = _make_job(tmp_path)
    (job_path / "translation.ru.json").write_text("{ not json")
    with pytest.raises(TranslationError, match="invalid translation"):
        subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_missing_segments_list(tmp_path):
    _make_job(tmp_path, translation={"version": 1, "step": "translate"})
    with pytest.raises(TranslationError, match="missing 'segments' list"):
        subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_missing_translated_text(tmp_path):
    translation = {**TRANSLATION, "segments": [{"id": 0, "start": 0.0, "end": 1.0}]}
    _make_job(tmp_path, translation=translation)
    with pytest.raises(TranslationError, match="missing 'translated_text'"):
        subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_empty_translated_text(tmp_path):
    translation = {**TRANSLATION, "segments": [{"id": 0, "start": 0.0, "end": 1.0, "translated_text": "   "}]}
    _make_job(tmp_path, translation=translation)
    with pytest.raises(TranslationError, match="empty translated_text"):
        subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_invalid_timestamp_type(tmp_path):
    translation = {**TRANSLATION, "segments": [{"id": 0, "start": "x", "end": 1.0, "translated_text": "t"}]}
    _make_job(tmp_path, translation=translation)
    with pytest.raises(TranslationError, match="non-numeric start"):
        subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_end_before_start(tmp_path):
    translation = {**TRANSLATION, "segments": [{"id": 0, "start": 2.0, "end": 1.0, "translated_text": "t"}]}
    _make_job(tmp_path, translation=translation)
    with pytest.raises(TranslationError, match="end <= start"):
        subtitles_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_reports_progress(tmp_path):
    _make_job(tmp_path)
    lines: list[str] = []
    subtitles_step.run("jobA", _config(tmp_path), echo=lines.append)

    assert any("wrote" in line and "subtitles.ru.srt" in line for line in lines)
