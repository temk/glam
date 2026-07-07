import os
import pytest
from types import SimpleNamespace
from pathlib import Path

import glam.media as media
from glam.steps import mux as mux_step
from glam.steps.mux import MuxError, _language_of, _build_command
from glam.common.job import JobInfo, Languages, SourceInfo, JobManifest, write_job_manifest
from glam.common.config import Config


def _make_job(tmp_path: Path, filename: str = "lecture.mov", tts=(), subs=(), with_source: bool = True) -> Path:
    job_path = tmp_path / "jobA"
    job_path.mkdir(parents=True)
    manifest = JobManifest(
        version=1,
        job=JobInfo(id="jobA", created_at="2026-07-05T00:00:00+00:00"),
        source=SourceInfo(
            original_path="/x/lecture.mov",
            filename=filename,
            artifact="source.mp4",
            audio_artifact="audio.wav",
            duration_seconds=1.0,
        ),
        languages=Languages(source="en", target="ru"),
    )
    write_job_manifest(manifest, job_path / "job.yaml")
    if with_source:
        (job_path / "source.mp4").write_bytes(b"src")
    for name in tts:
        (job_path / name).write_bytes(b"wav")
    for name in subs:
        (job_path / name).write_text("srt")
    return job_path


def _config(tmp_path: Path) -> Config:
    return Config(services=[], job_dir=tmp_path)


@pytest.fixture
def ffmpeg(monkeypatch):
    """Fake media.run_ffmpeg (records args, writes the output) and media.ffprobe_json."""
    calls: list[list[str]] = []
    state = {"fail": False, "has_audio": True}

    def run(args, error_prefix, error_cls=media.MediaError):
        calls.append([str(a) for a in args])
        if state["fail"]:
            raise error_cls(f"{error_prefix}: boom")
        Path(args[-1]).write_bytes(b"fake-mp4")

    def probe(path):
        streams = [{"codec_type": "video"}]
        if state["has_audio"]:
            streams.append({"codec_type": "audio"})
        return {"streams": streams}

    monkeypatch.setattr(media, "run_ffmpeg", run)
    monkeypatch.setattr(media, "ffprobe_json", probe)
    return SimpleNamespace(calls=calls, state=state)


def _inputs(args: list[str]) -> list[str]:
    return [Path(args[i + 1]).name for i, a in enumerate(args) if a == "-i"]


def _maps(args: list[str]) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == "-map"]


# --- run() ---


def test_creates_result_and_output_name(tmp_path, ffmpeg):
    job = _make_job(tmp_path, filename="lecture.mov", tts=["tts.ru.wav"])
    path = mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert path.name == "lecture.mp4"
    assert path.exists()
    assert (job / "result.mp4").is_symlink()
    assert os.readlink(job / "result.mp4") == "lecture.mp4"


def test_discovers_and_sorts_artifacts(tmp_path, ffmpeg):
    _make_job(tmp_path, tts=["tts.ru.wav", "tts.de.wav"], subs=["subtitles.ru.srt", "subtitles.de.srt"])
    mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    # source first, then tts and subs sorted by filename.
    assert _inputs(ffmpeg.calls[0]) == [
        "source.mp4",
        "tts.de.wav",
        "tts.ru.wav",
        "subtitles.de.srt",
        "subtitles.ru.srt",
    ]


def test_adds_all_tracks_and_language_tags(tmp_path, ffmpeg):
    _make_job(tmp_path, tts=["tts.de.wav", "tts.ru.wav"], subs=["subtitles.de.srt"])
    mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
    args = ffmpeg.calls[0]

    # video + source audio + 2 tts + 1 subtitle
    assert _maps(args) == ["0:v:0", "0:a:0", "1:a:0", "2:a:0", "3:0"]
    # source audio copied, tts encoded
    assert "-c:v" in args and "-c:s" in args
    assert args[args.index("-c:a:0") + 1] == "copy"
    assert args[args.index("-c:a:1") + 1] == "aac"
    # language tags: source (en), tts de/ru, subtitle de
    assert args[args.index("-metadata:s:a:0") + 1] == "language=en"
    assert args[args.index("-metadata:s:a:1") + 1] == "language=de"
    assert args[args.index("-metadata:s:a:2") + 1] == "language=ru"
    assert args[args.index("-metadata:s:s:0") + 1] == "language=de"


def test_exclude_audio(tmp_path, ffmpeg):
    _make_job(tmp_path, tts=["tts.ru.wav", "tts.de.wav"])
    mux_step.run("jobA", _config(tmp_path), exclude=("tts.de.wav",), echo=lambda *_: None)

    assert _inputs(ffmpeg.calls[0]) == ["source.mp4", "tts.ru.wav"]


def test_exclude_subtitle(tmp_path, ffmpeg):
    _make_job(tmp_path, subs=["subtitles.ru.srt", "subtitles.en.srt"])
    mux_step.run("jobA", _config(tmp_path), exclude=("subtitles.en.srt",), echo=lambda *_: None)

    assert _inputs(ffmpeg.calls[0]) == ["source.mp4", "subtitles.ru.srt"]


def test_exclude_nonexisting_is_error(tmp_path, ffmpeg):
    _make_job(tmp_path, tts=["tts.ru.wav"])
    with pytest.raises(MuxError, match="does not exist"):
        mux_step.run("jobA", _config(tmp_path), exclude=("tts.zz.wav",), echo=lambda *_: None)


def test_exclude_invalid_kind_is_error(tmp_path, ffmpeg):
    _make_job(tmp_path)
    (tmp_path / "jobA" / "notes.txt").write_text("x")
    with pytest.raises(MuxError, match="is not a"):
        mux_step.run("jobA", _config(tmp_path), exclude=("notes.txt",), echo=lambda *_: None)


def test_leaves_input_artifacts_unchanged(tmp_path, ffmpeg):
    job = _make_job(tmp_path, tts=["tts.ru.wav"], subs=["subtitles.ru.srt"])
    mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert (job / "tts.ru.wav").read_bytes() == b"wav"
    assert (job / "subtitles.ru.srt").read_text() == "srt"
    assert (job / "source.mp4").read_bytes() == b"src"


def test_skips_when_result_exists(tmp_path, ffmpeg):
    job = _make_job(tmp_path, tts=["tts.ru.wav"])
    (job / "lecture.mp4").write_bytes(b"old")
    os.symlink("lecture.mp4", job / "result.mp4")

    mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
    assert ffmpeg.calls == []  # not rebuilt
    assert (job / "lecture.mp4").read_bytes() == b"old"


def test_restores_missing_result_without_rebuild(tmp_path, ffmpeg):
    job = _make_job(tmp_path, tts=["tts.ru.wav"])
    (job / "lecture.mp4").write_bytes(b"old")  # output exists, result.mp4 missing

    mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
    assert ffmpeg.calls == []  # not rebuilt
    assert os.readlink(job / "result.mp4") == "lecture.mp4"


def test_restores_incorrect_result_without_rebuild(tmp_path, ffmpeg):
    job = _make_job(tmp_path, tts=["tts.ru.wav"])
    (job / "lecture.mp4").write_bytes(b"old")
    os.symlink("other.mp4", job / "result.mp4")  # points elsewhere

    mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
    assert ffmpeg.calls == []
    assert os.readlink(job / "result.mp4") == "lecture.mp4"


def test_force_rebuilds(tmp_path, ffmpeg):
    job = _make_job(tmp_path, tts=["tts.ru.wav"])
    (job / "lecture.mp4").write_bytes(b"old")
    os.symlink("lecture.mp4", job / "result.mp4")

    mux_step.run("jobA", _config(tmp_path), force=True, echo=lambda *_: None)
    assert len(ffmpeg.calls) == 1
    assert (job / "lecture.mp4").read_bytes() == b"fake-mp4"


def test_missing_source_is_error(tmp_path, ffmpeg):
    _make_job(tmp_path, with_source=False)
    with pytest.raises(MuxError, match="missing source video"):
        mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_ffmpeg_failure_is_error(tmp_path, ffmpeg):
    _make_job(tmp_path, tts=["tts.ru.wav"])
    ffmpeg.state["fail"] = True
    with pytest.raises(MuxError):
        mux_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_job_not_found(tmp_path, ffmpeg):
    with pytest.raises(MuxError, match="job not found"):
        mux_step.run("missing", _config(tmp_path), echo=lambda *_: None)


# --- helpers ---


@pytest.mark.parametrize(
    "name, expected",
    [
        ("tts.ru.wav", "ru"),
        ("tts.ru.Abigail.wav", "ru"),
        ("subtitles.en.srt", "en"),
        ("tts.wav", None),
        ("result.mp4", None),
    ],
)
def test_language_of(name, expected):
    assert _language_of(name) == expected


def test_build_command_without_source_audio(tmp_path):
    args = _build_command(Path("source.mp4"), "en", [Path("tts.ru.wav")], [], Path("out.mp4"), has_source_audio=False)
    assert _maps(args) == ["0:v:0", "1:a:0"]  # no source audio map
    assert args[args.index("-c:a:0") + 1] == "aac"  # first output audio is the tts track
    assert "-metadata:s:a:0" in args and args[args.index("-metadata:s:a:0") + 1] == "language=ru"
