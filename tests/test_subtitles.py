import json

import pysubs2
import pytest

from glam.errors import GlamError
from glam.steps import subtitles


def write_translation(job_dir, lang="ru", model="fake-model", segments=None):
    job_dir.mkdir(parents=True, exist_ok=True)
    segments = segments or [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "привет"},
        {"id": 1, "start": 2.0, "end": 4.0, "text": "мир"},
    ]
    path = job_dir / f"translation.{lang}.{model}.json"
    path.write_text(json.dumps({"model": model, "lang": lang, "segments": segments}))
    return path


def test_subtitles_writes_srt_with_model_slug_from_translation(tmp_path):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir, model="qwen2-5-7b")

    output_path = subtitles.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)

    assert output_path == job_dir / "subtitles.ru.qwen2-5-7b.srt"
    assert output_path.exists()

    subs = pysubs2.load(str(output_path))
    assert len(subs) == 2
    assert subs[0].plaintext == "привет"
    assert subs[1].plaintext == "мир"


def test_subtitles_is_idempotent_unless_forced(tmp_path):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir)
    output_path = job_dir / "subtitles.ru.fake-model.srt"
    output_path.write_text("already there")

    result_path = subtitles.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)

    assert result_path == output_path
    assert output_path.read_text() == "already there"


def test_subtitles_force_recomputes(tmp_path):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir)
    output_path = job_dir / "subtitles.ru.fake-model.srt"
    output_path.write_text("stale")

    subtitles.run("myvid", "ru", jobs_root=tmp_path / "jobs", force=True, echo=lambda *_: None)

    assert "stale" not in output_path.read_text()


def test_subtitles_missing_translation_raises(tmp_path):
    (tmp_path / "jobs" / "myvid").mkdir(parents=True)
    with pytest.raises(GlamError):
        subtitles.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)


def test_subtitles_ambiguous_translations_raises(tmp_path):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir, model="model-a")
    write_translation(job_dir, model="model-b")
    with pytest.raises(GlamError):
        subtitles.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)


def test_subtitles_explicit_translation_path(tmp_path):
    job_dir = tmp_path / "jobs" / "myvid"
    write_translation(job_dir, model="model-a")
    explicit = write_translation(job_dir, model="model-b")

    output_path = subtitles.run(
        "myvid", "ru", jobs_root=tmp_path / "jobs",
        translation_path=explicit, echo=lambda *_: None,
    )
    assert output_path == job_dir / "subtitles.ru.model-b.srt"


def test_subtitles_reading_speed_resegments_long_text(tmp_path):
    job_dir = tmp_path / "jobs" / "myvid"
    long_text = "слово " * 40  # far more than fits in the 1s window at default cps
    write_translation(job_dir, segments=[
        {"id": 0, "start": 0.0, "end": 1.0, "text": long_text.strip()},
    ])

    output_path = subtitles.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)
    subs = pysubs2.load(str(output_path))

    assert len(subs) > 1
    for event in subs:
        assert len(event.plaintext.split("\n")) <= 2
