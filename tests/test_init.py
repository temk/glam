import subprocess
from pathlib import Path

from glam.steps import init as init_step


def make_sample_video(path, title="Sample Title"):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-c:a", "aac",
            "-metadata", f"title={title}",
            str(path),
        ],
        capture_output=True,
        check=True,
    )


def test_init_creates_job_artifacts(tmp_path):
    video = tmp_path / "input.mp4"
    make_sample_video(video)
    jobs_root = tmp_path / "jobs"

    job = init_step.run(video, jobs_root=jobs_root, echo=lambda *_: None)

    assert job.video_id == "input"
    assert job.job_dir == jobs_root / "input"
    assert (job.job_dir / "source.mp4").resolve() == video.resolve()
    assert (job.job_dir / "audio.wav").exists()

    meta = (job.job_dir / "meta.json").read_text()
    assert "Sample Title" in meta
    assert "input.mp4" in meta


def test_init_is_idempotent_unless_forced(tmp_path):
    video = tmp_path / "input.mp4"
    make_sample_video(video)
    jobs_root = tmp_path / "jobs"

    init_step.run(video, jobs_root=jobs_root, echo=lambda *_: None)
    audio_path = jobs_root / "input" / "audio.wav"
    first_mtime = audio_path.stat().st_mtime_ns

    init_step.run(video, jobs_root=jobs_root, echo=lambda *_: None)
    assert audio_path.stat().st_mtime_ns == first_mtime

    init_step.run(video, jobs_root=jobs_root, force=True, echo=lambda *_: None)
    assert audio_path.stat().st_mtime_ns != first_mtime


def test_init_custom_video_id(tmp_path):
    video = tmp_path / "input.mp4"
    make_sample_video(video)
    jobs_root = tmp_path / "jobs"

    job = init_step.run(video, video_id="custom-id", jobs_root=jobs_root, echo=lambda *_: None)

    assert job.video_id == "custom-id"
    assert job.job_dir == jobs_root / "custom-id"
