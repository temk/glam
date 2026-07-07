import json
import pytest
from pathlib import Path

from glam.steps import accent as accent_step
from glam.common.job import JobInfo, Languages, SourceInfo, JobManifest, write_job_manifest
from glam.steps.accent import STRESS_MARK, AccentError, _plus_to_combining, _fix_russian_stress
from glam.common.config import Config

TRANSLATION = {
    "version": 1,
    "step": "translate",
    "job_id": "jobA",
    "source_language": "en",
    "model": "test-model",
    "audio_artifact": "audio.wav",
    "segments": [
        {"id": 1, "start": 0.0, "end": 1.0, "text": "Hello.", "translated_text": "Привет."},
        {"id": 2, "start": 1.0, "end": 2.0, "text": "World.", "translated_text": "Мир."},
    ],
}


def _make_job(tmp_path: Path, target: str = "ru", translation: dict | None = None) -> Path:
    job_path = tmp_path / "jobA"
    job_path.mkdir(parents=True)
    manifest = JobManifest(
        version=1,
        job=JobInfo(id="jobA", created_at="2026-07-05T00:00:00+00:00"),
        source=SourceInfo(
            original_path="/x/lecture.mov",
            filename="lecture.mov",
            artifact="source.mp4",
            audio_artifact="audio.wav",
            duration_seconds=1.0,
        ),
        languages=Languages(source="en", target=target),
    )
    write_job_manifest(manifest, job_path / "job.yaml")
    if translation is not None:
        (job_path / f"translation.{target}.json").write_text(json.dumps(translation, ensure_ascii=False))
    return job_path


def _config(tmp_path: Path) -> Config:
    return Config(services=[], job_dir=tmp_path)


@pytest.fixture
def fake_ru_accentor(monkeypatch):
    """Replace the heavy silero loader with a fake callable so tests never import torch."""

    def accentor(sentence: str, **kwargs) -> str:
        # Emulate silero: write `+` before the first vowel of each word.
        out = []
        for word in sentence.split(" "):
            marked = False
            for ch in word:
                if not marked and ch.lower() in "аеёиоуыэюя":
                    out.append("+")
                    marked = True
                out.append(ch)
            out.append(" ")
        return "".join(out).rstrip(" ")

    monkeypatch.setattr(accent_step, "_load_ru_accentor", lambda: accentor)


# --- run() ---


def test_writes_fixed_file_for_russian(tmp_path, fake_ru_accentor):
    _make_job(tmp_path, target="ru", translation=TRANSLATION)
    path = accent_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert path is not None and path.name == "translation.ru.fixed.json"
    doc = json.loads(path.read_text())
    assert STRESS_MARK in doc["segments"][0]["translated_text"]  # accent applied
    assert "+" not in doc["segments"][0]["translated_text"]  # silero's '+' converted, not left in


def test_preserves_structure_and_source_text(tmp_path, fake_ru_accentor):
    _make_job(tmp_path, target="ru", translation=TRANSLATION)
    path = accent_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
    doc = json.loads(path.read_text())

    assert {k: v for k, v in doc.items() if k != "segments"} == {
        k: v for k, v in TRANSLATION.items() if k != "segments"
    }
    seg = doc["segments"][0]
    assert seg["id"] == 1 and seg["start"] == 0.0 and seg["end"] == 1.0
    assert seg["text"] == "Hello."  # English source untouched


def test_noop_for_language_without_fixer(tmp_path):
    _make_job(tmp_path, target="de", translation={**TRANSLATION, "segments": []})
    logs: list[str] = []
    result = accent_step.run("jobA", _config(tmp_path), target="de", echo=logs.append)

    assert result is None
    assert not (tmp_path / "jobA" / "translation.de.fixed.json").exists()
    assert any("no text fixer" in line for line in logs)


def test_skips_when_fixed_exists(tmp_path, fake_ru_accentor):
    job = _make_job(tmp_path, target="ru", translation=TRANSLATION)
    (job / "translation.ru.fixed.json").write_text('{"kept": true}')

    accent_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
    assert json.loads((job / "translation.ru.fixed.json").read_text()) == {"kept": True}  # not overwritten


def test_force_rebuilds(tmp_path, fake_ru_accentor):
    job = _make_job(tmp_path, target="ru", translation=TRANSLATION)
    (job / "translation.ru.fixed.json").write_text('{"kept": true}')

    accent_step.run("jobA", _config(tmp_path), force=True, echo=lambda *_: None)
    doc = json.loads((job / "translation.ru.fixed.json").read_text())
    assert "segments" in doc  # rebuilt from the source translation


def test_missing_translation_is_error(tmp_path, fake_ru_accentor):
    _make_job(tmp_path, target="ru", translation=None)  # no translation.ru.json
    with pytest.raises(AccentError, match="missing translation artifact"):
        accent_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_job_not_found(tmp_path):
    with pytest.raises(AccentError, match="job not found"):
        accent_step.run("missing", _config(tmp_path), echo=lambda *_: None)


# --- Russian fixer transform ---


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("зов+ут", "зов" + "у" + STRESS_MARK + "т"),  # '+' before vowel -> combining accent after it
        ("+я", "я" + STRESS_MARK),
        ("Л+ёва", "Л" + "ё" + STRESS_MARK + "ва"),
        ("без ударения", "без ударения"),  # no '+' -> unchanged
    ],
)
def test_plus_to_combining(raw, expected):
    assert _plus_to_combining(raw) == expected
    assert "+" not in _plus_to_combining(raw)


def test_fix_russian_marks_stress(fake_ru_accentor):
    out = _fix_russian_stress(["Привет", ""])
    assert STRESS_MARK in out[0] and "+" not in out[0]
    assert out[1] == ""  # empty text is left untouched
