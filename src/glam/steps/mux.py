from pathlib import Path

import ffmpeg

from glam.errors import GlamError
from glam.paths import job_dir, resolve_artifact

HARDSUB_CRF = 18
HARDSUB_PRESET = "medium"


class MuxError(GlamError):
    pass


def run(video_id, lang, jobs_root=Path("jobs"), hardsub=False, keep_original_audio=True,
        subtitles_path=None, tts_track_path=None, force=False, echo=print):
    job_path = job_dir(jobs_root, video_id)
    source_path = resolve_artifact(job_path, "source.*", hint="run 'glam init' first")
    tts_track_path = resolve_artifact(
        job_path, f"tts_track.{lang}.*.wav", tts_track_path,
        hint=f"run 'glam tts --lang {lang}' first",
    )

    output_path = job_path / f"output.{lang}.mkv"
    if output_path.exists() and not force:
        echo(f"skip mux, already exists: {output_path}")
        return output_path

    video_in = ffmpeg.input(str(source_path))
    dub_in = ffmpeg.input(str(tts_track_path))
    video_stream = video_in.video

    streams = []
    kwargs = {}

    if hardsub:
        subtitles_path = resolve_artifact(
            job_path, f"subtitles.{lang}.*.srt", subtitles_path,
            hint=f"run 'glam subtitles --lang {lang}' first",
        )
        video_stream = video_stream.filter("subtitles", str(subtitles_path))
        kwargs["vcodec"] = "libx264"
        kwargs["crf"] = HARDSUB_CRF
        kwargs["preset"] = HARDSUB_PRESET
    else:
        kwargs["vcodec"] = "copy"

    streams.append(video_stream)

    audio_idx = 0
    if keep_original_audio:
        streams.append(video_in.audio)
        kwargs[f"c:a:{audio_idx}"] = "copy"
        audio_idx += 1
    streams.append(dub_in.audio)
    kwargs[f"c:a:{audio_idx}"] = "aac"

    if not hardsub:
        subtitles_path = resolve_artifact(
            job_path, f"subtitles.{lang}.*.srt", subtitles_path,
            hint=f"run 'glam subtitles --lang {lang}' first",
        )
        sub_in = ffmpeg.input(str(subtitles_path))
        streams.append(sub_in["s"])
        kwargs["c:s"] = "srt"

    try:
        ffmpeg.output(*streams, str(output_path), **kwargs).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise MuxError(f"mux failed: {stderr.strip()}") from e

    echo(f"wrote {output_path}")
    return output_path
