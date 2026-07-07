from pathlib import Path
from dataclasses import dataclass

from glam.common.job import JOB_MANIFEST_NAME, read_job_manifest
from glam.common.config import Config
from glam.common.errors import GlamError
from glam.common.translation import translation_filename, load_translated_segments

SUBTITLES_NAME_TEMPLATE = "subtitles.{}.srt"

# Soft wrap width for a single subtitle line; long cues are split into readable lines.
MAX_LINE_LENGTH = 42


class SubtitlesError(GlamError):
    pass


@dataclass
class Cue:
    start: float
    end: float
    text: str


def run(job_id: str, config: Config, target: str | None = None, force: bool = False, echo=print) -> Path:
    """Render a job's translated segments into an SRT subtitle file."""
    job_path = config.job_dir / job_id
    if not job_path.is_dir():
        raise SubtitlesError(f"job not found: {job_id} (looked in {job_path})")

    manifest = read_job_manifest(job_path / JOB_MANIFEST_NAME)
    target = target or manifest.languages.target
    if not target:
        raise SubtitlesError("missing target language: pass --target or set languages.target in job.yaml")

    subtitles_path = job_path / SUBTITLES_NAME_TEMPLATE.format(target)
    if subtitles_path.exists() and not force:
        echo(f"skip subtitles, already exists: {subtitles_path}")
        return subtitles_path

    segments = load_translated_segments(job_path / translation_filename(target))
    cues = [Cue(start=s.start, end=s.end, text=_wrap_text(s.translated_text)) for s in segments]

    subtitles_path.write_text(_render_srt(cues), encoding="utf-8")
    echo(f"wrote {subtitles_path} ({len(cues)} cues)")
    return subtitles_path


def _wrap_text(text: str, width: int = MAX_LINE_LENGTH) -> str:
    """Greedy word wrap so no line exceeds `width`; a word longer than `width` stays whole."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _render_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{index}\n{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n{cue.text}"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n\n".join(blocks) + "\n"


def _format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
