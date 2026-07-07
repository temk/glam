import os
from fnmatch import fnmatch
from pathlib import Path

from glam import media
from glam.common.job import JOB_MANIFEST_NAME, read_job_manifest
from glam.common.config import Config
from glam.common.errors import GlamError

RESULT_NAME = "result.mp4"
TTS_GLOB = "tts*.wav"
SUBTITLES_GLOB = "subtitles*.srt"


class MuxError(GlamError):
    pass


def run(job_id: str, config: Config, exclude: tuple[str, ...] = (), force: bool = False, echo=print) -> Path:
    """Mux the source video with all discovered TTS and subtitle artifacts into the final MP4."""
    job_path = config.job_dir / job_id
    if not job_path.is_dir():
        raise MuxError(f"job not found: {job_id} (looked in {job_path})")

    manifest = read_job_manifest(job_path / JOB_MANIFEST_NAME)
    source_path = job_path / manifest.source.artifact
    if not source_path.exists():
        raise MuxError(f"missing source video: {source_path}")

    tts_wavs = sorted(job_path.glob(TTS_GLOB))
    subtitles = sorted(job_path.glob(SUBTITLES_GLOB))
    _validate_exclusions(exclude, job_path)
    tts_wavs = [w for w in tts_wavs if w.name not in exclude]
    subtitles = [s for s in subtitles if s.name not in exclude]

    output_path = job_path / (Path(manifest.source.filename).stem + ".mp4")
    result_path = job_path / RESULT_NAME

    if output_path.exists() and not force:
        if _links_to(result_path, output_path.name):
            echo(f"skip mux, already built: {output_path}")
        else:
            _link_result(result_path, output_path.name)
            echo(f"restored {result_path} -> {output_path.name}")
        return output_path

    args = _build_command(
        source_path, manifest.languages.source, tts_wavs, subtitles, output_path, _has_audio(source_path)
    )
    media.run_ffmpeg(args, f"ffmpeg mux failed for job {job_id}", error_cls=MuxError)
    _link_result(result_path, output_path.name)
    echo(f"wrote {output_path} ({len(tts_wavs)} audio, {len(subtitles)} subtitle tracks)")
    return output_path


def _validate_exclusions(exclude: tuple[str, ...], job_path: Path) -> None:
    for name in exclude:
        if not (fnmatch(name, TTS_GLOB) or fnmatch(name, SUBTITLES_GLOB)):
            raise MuxError(f"--exclude '{name}' is not a {TTS_GLOB} or {SUBTITLES_GLOB} artifact")
        if not (job_path / name).exists():
            raise MuxError(f"--exclude '{name}' does not exist in the job directory")


def _has_audio(source: Path) -> bool:
    info = media.ffprobe_json(source)
    return any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))


def _language_of(name: str) -> str | None:
    """Language token from an artifact filename: `tts.<lang>[.<voice>].wav`, `subtitles.<lang>.srt`."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in ("tts", "subtitles"):
        return parts[1]
    return None


def _build_command(
    source: Path,
    source_language: str,
    tts_wavs: list[Path],
    subtitles: list[Path],
    output: Path,
    has_source_audio: bool,
) -> list[str]:
    args = ["-i", str(source)]
    for wav in tts_wavs:
        args += ["-i", str(wav)]
    for srt in subtitles:
        args += ["-i", str(srt)]

    args += ["-map", "0:v:0"]
    if has_source_audio:
        args += ["-map", "0:a:0"]
    for i in range(len(tts_wavs)):
        args += ["-map", f"{i + 1}:a:0"]
    subtitle_input_base = 1 + len(tts_wavs)
    for j in range(len(subtitles)):
        args += ["-map", f"{subtitle_input_base + j}:0"]

    args += ["-c:v", "copy", "-c:s", "mov_text"]

    # Output audio streams follow map order: optional source audio first (copied), then each
    # synthesized track (encoded to AAC). Language metadata is indexed to that same order.
    audio_index = 0
    if has_source_audio:
        args += ["-c:a:0", "copy"]
        if source_language:
            args += ["-metadata:s:a:0", f"language={source_language}"]
        audio_index = 1
    for k, wav in enumerate(tts_wavs):
        index = audio_index + k
        args += [f"-c:a:{index}", "aac"]
        language = _language_of(wav.name)
        if language:
            args += [f"-metadata:s:a:{index}", f"language={language}"]

    for j, srt in enumerate(subtitles):
        language = _language_of(srt.name)
        if language:
            args += [f"-metadata:s:s:{j}", f"language={language}"]

    args += [str(output)]
    return args


def _links_to(link: Path, target_name: str) -> bool:
    return link.is_symlink() and os.readlink(link) == target_name


def _link_result(link: Path, target_name: str) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target_name)
