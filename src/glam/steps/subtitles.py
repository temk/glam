import json
from pathlib import Path

import pysubs2

from glam.errors import GlamError
from glam.paths import job_dir, resolve_artifact
from glam.resegment import (
    DEFAULT_CPS,
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MAX_LINES,
    DEFAULT_MIN_GAP,
    resegment,
)


class SubtitlesError(GlamError):
    pass


def run(video_id, lang, jobs_root=Path("jobs"), translation_path=None, force=False,
        cps=DEFAULT_CPS, max_chars_per_line=DEFAULT_MAX_CHARS_PER_LINE,
        max_lines=DEFAULT_MAX_LINES, min_gap=DEFAULT_MIN_GAP, echo=print):
    job_path = job_dir(jobs_root, video_id)

    translation_path = resolve_artifact(
        job_path, f"translation.{lang}.*.json", translation_path,
        hint=f"run 'glam translate --lang {lang}' first",
    )
    model_slug = _model_slug(translation_path)
    output_path = job_path / f"subtitles.{lang}.{model_slug}.srt"
    if output_path.exists() and not force:
        echo(f"skip subtitles, already exists: {output_path}")
        return output_path

    translation = json.loads(Path(translation_path).read_text())
    segments = translation.get("segments") or []
    if not segments:
        raise SubtitlesError(f"no segments found in {translation_path}")

    cues = resegment(
        segments, cps=cps, max_chars_per_line=max_chars_per_line,
        max_lines=max_lines, min_gap=min_gap,
    )
    if not cues:
        raise SubtitlesError(f"resegmentation produced no cues from {translation_path}")

    subs = pysubs2.SSAFile()
    for cue in cues:
        subs.append(pysubs2.SSAEvent(
            start=round(cue.start * 1000),
            end=round(cue.end * 1000),
            text=cue.text.replace("\n", r"\N"),
        ))
    subs.save(str(output_path))
    echo(f"wrote {output_path}")
    return output_path


def _model_slug(translation_path):
    parts = Path(translation_path).stem.split(".")
    return parts[2] if len(parts) > 2 else "unknown"
