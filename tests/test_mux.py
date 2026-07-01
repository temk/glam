import ffmpeg
import pytest

from glam import media
from glam.errors import GlamError
from glam.steps import mux


def make_video(path, duration=2, size="64x64"):
    video_in = ffmpeg.input(f"testsrc=duration={duration}:size={size}:rate=5", f="lavfi")
    audio_in = ffmpeg.input(f"sine=frequency=440:duration={duration}", f="lavfi")
    (
        ffmpeg
        .output(video_in, audio_in, str(path), vcodec="libx264", acodec="aac", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )


def make_tone_wav(path, duration=2, sample_rate=22050):
    (
        ffmpeg
        .input(f"sine=frequency=220:duration={duration}", f="lavfi")
        .output(str(path), format="wav", ar=sample_rate, ac=1, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )


def make_srt(path):
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n")


def make_job(tmp_path, video_id="myvid", lang="ru"):
    job_dir = tmp_path / "jobs" / video_id
    job_dir.mkdir(parents=True)
    make_video(job_dir / "source.mp4")
    make_tone_wav(job_dir / f"tts_track.{lang}.fake-tts.default.wav")
    make_srt(job_dir / f"subtitles.{lang}.fake-model.srt")
    return job_dir


def probe_streams(path):
    return ffmpeg.probe(str(path))["streams"]


def test_mux_soft_subs_default(tmp_path):
    job_dir = make_job(tmp_path)

    output_path = mux.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)

    assert output_path == job_dir / "output.ru.mkv"
    streams = probe_streams(output_path)
    codec_types = [s["codec_type"] for s in streams]
    assert codec_types.count("video") == 1
    assert codec_types.count("audio") == 2  # original + dubbed
    assert codec_types.count("subtitle") == 1


def test_mux_without_original_audio(tmp_path):
    job_dir = make_job(tmp_path)

    output_path = mux.run(
        "myvid", "ru", jobs_root=tmp_path / "jobs",
        keep_original_audio=False, echo=lambda *_: None,
    )

    streams = probe_streams(output_path)
    assert [s["codec_type"] for s in streams].count("audio") == 1


def test_mux_hardsub_has_no_subtitle_stream(tmp_path):
    job_dir = make_job(tmp_path)

    output_path = mux.run(
        "myvid", "ru", jobs_root=tmp_path / "jobs",
        hardsub=True, echo=lambda *_: None,
    )

    streams = probe_streams(output_path)
    assert [s["codec_type"] for s in streams].count("subtitle") == 0
    assert [s["codec_type"] for s in streams].count("video") == 1


def test_mux_is_idempotent_unless_forced(tmp_path):
    job_dir = make_job(tmp_path)
    output_path = job_dir / "output.ru.mkv"

    mux.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)
    first_mtime = output_path.stat().st_mtime_ns

    mux.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)
    assert output_path.stat().st_mtime_ns == first_mtime

    mux.run("myvid", "ru", jobs_root=tmp_path / "jobs", force=True, echo=lambda *_: None)
    assert output_path.stat().st_mtime_ns != first_mtime


def test_mux_missing_source_raises(tmp_path):
    job_dir = tmp_path / "jobs" / "myvid"
    job_dir.mkdir(parents=True)
    make_tone_wav(job_dir / "tts_track.ru.fake-tts.default.wav")
    make_srt(job_dir / "subtitles.ru.fake-model.srt")

    with pytest.raises(GlamError):
        mux.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)


def test_mux_missing_tts_track_raises(tmp_path):
    job_dir = tmp_path / "jobs" / "myvid"
    job_dir.mkdir(parents=True)
    make_video(job_dir / "source.mp4")
    make_srt(job_dir / "subtitles.ru.fake-model.srt")

    with pytest.raises(GlamError):
        mux.run("myvid", "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)
