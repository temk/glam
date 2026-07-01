import json
from dataclasses import dataclass
from pathlib import Path

from glam import media
from glam.paths import job_dir, slugify


@dataclass
class Job:
    video_id: str
    job_dir: Path


def run(video_file, video_id=None, jobs_root=Path("jobs"), force=False, echo=print):
    video_file = Path(video_file).resolve()
    if video_id is None:
        video_id = slugify(video_file.stem)

    job_path = job_dir(jobs_root, video_id)
    job_path.mkdir(parents=True, exist_ok=True)

    source_link = job_path / f"source{video_file.suffix}"
    if force or not source_link.exists():
        if source_link.is_symlink() or source_link.exists():
            source_link.unlink()
        source_link.symlink_to(video_file)
        echo(f"linked {source_link} -> {video_file}")
    else:
        echo(f"skip source link, already exists: {source_link}")

    meta_path = job_path / "meta.json"
    if force or not meta_path.exists():
        probe = media.ffprobe_json(video_file)
        fmt = probe.get("format", {})
        title = fmt.get("tags", {}).get("title") or video_file.stem
        duration = float(fmt["duration"]) if "duration" in fmt else None
        meta = {
            "title": title,
            "duration": duration,
            "source_filename": video_file.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        echo(f"wrote {meta_path}")
    else:
        echo(f"skip metadata, already exists: {meta_path}")

    audio_path = job_path / "audio.wav"
    if force or not audio_path.exists():
        media.extract_audio(video_file, audio_path)
        echo(f"wrote {audio_path}")
    else:
        echo(f"skip audio extraction, already exists: {audio_path}")

    return Job(video_id=video_id, job_dir=job_path)
