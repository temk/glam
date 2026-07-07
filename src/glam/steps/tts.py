import io
import wave
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from glam.common.job import JOB_MANIFEST_NAME, read_job_manifest
from glam.common.config import Config, ServiceName, ServiceConfig
from glam.common.errors import GlamError
from glam.backend.tts.base import build_tts_backend
from glam.common.translation import (
    TranslatedSegment,
    translation_filename,
    load_translated_segments,
    fixed_translation_filename,
)

# Artifact is named after the target language; a chosen voice also enters the name
# because it changes the result (see docs/architecture.md "Artifact names").
TTS_NAME_TEMPLATE = "tts.{}.wav"
TTS_VOICE_NAME_TEMPLATE = "tts.{}.{}.wav"


class TtsError(GlamError):
    pass


@dataclass
class _Fragment:
    nchannels: int
    sampwidth: int
    framerate: int
    frames: bytes

    @property
    def nframes(self) -> int:
        return len(self.frames) // (self.sampwidth * self.nchannels)


def run(
    job_id: str, config: Config, target: str | None = None, voice: str | None = None, force: bool = False, echo=print
) -> Path:
    """Synthesize a dubbed target-language audio track from a job's translated segments."""
    job_path = config.job_dir / job_id
    if not job_path.is_dir():
        raise TtsError(f"job not found: {job_id} (looked in {job_path})")

    manifest = read_job_manifest(job_path / JOB_MANIFEST_NAME)
    target = target or manifest.languages.target
    if not target:
        raise TtsError("missing target language: pass --target or set languages.target in job.yaml")
    # `--voice` overrides the job's default voice; when neither is set the voice is left unset so
    # the backend picks its own default, and the artifact name carries no voice.
    voice = voice or manifest.voice

    name = TTS_VOICE_NAME_TEMPLATE.format(target, voice) if voice else TTS_NAME_TEMPLATE.format(target)
    output_path = job_path / name
    if output_path.exists() and not force:
        echo(f"skip tts, already exists: {output_path}")
        return output_path

    segments = load_translated_segments(_translation_source(job_path, target, echo))
    service = config[ServiceName.TTS]
    fragments = _synthesize(segments, service, target, voice, echo)
    _assemble(output_path, segments, fragments)

    echo(f"wrote {output_path} ({len(segments)} segments)")
    return output_path


def _translation_source(job_path: Path, target: str, echo) -> Path:
    """Prefer the `accent` step's corrected translation when it exists, else the plain one."""
    fixed = job_path / fixed_translation_filename(target)
    if fixed.exists():
        echo(f"using corrected translation: {fixed.name}")
        return fixed
    return job_path / translation_filename(target)


def _synthesize(
    segments: list[TranslatedSegment], service: ServiceConfig, target: str, voice: str | None, echo
) -> list[_Fragment]:
    backend = build_tts_backend(service)
    total = len(segments)
    fragments = []
    for index, segment in enumerate(segments, start=1):
        echo(f"[{datetime.now():%H:%M:%S}] synthesizing segment {index}/{total}")
        audio = backend.synthesize(segment.translated_text, target=target, voice=voice)
        fragments.append(_read_wav(audio, segment))
    return fragments


def _read_wav(data: bytes, segment: TranslatedSegment) -> _Fragment:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return _Fragment(
                nchannels=w.getnchannels(),
                sampwidth=w.getsampwidth(),
                framerate=w.getframerate(),
                frames=w.readframes(w.getnframes()),
            )
    except (wave.Error, EOFError) as e:
        raise TtsError(f"TTS service returned an invalid response for segment {segment.id}: {e}") from e


def _assemble(output_path: Path, segments: list[TranslatedSegment], fragments: list[_Fragment]) -> None:
    if not fragments:
        raise TtsError("unable to assemble audio: no segments to synthesize")
    ref = fragments[0]
    for fragment, segment in zip(fragments, segments):
        if (fragment.nchannels, fragment.sampwidth, fragment.framerate) != (
            ref.nchannels,
            ref.sampwidth,
            ref.framerate,
        ):
            raise TtsError(
                f"unable to assemble audio: segment {segment.id} has a different audio format than the first segment"
            )

    silence_frame = b"\x00" * (ref.sampwidth * ref.nchannels)
    track = bytearray()
    written_frames = 0  # frames already placed on the track
    for fragment, segment in zip(fragments, segments):
        start_frame = round(segment.start * ref.framerate)
        # Anchor the fragment at its `start`; a gap before it is padded with silence. When the
        # previous fragment overran past this `start`, we do NOT trim — we append right after it,
        # pushing this segment (and the rest) later. Timing drifts but no audio is lost.
        if start_frame > written_frames:
            gap = start_frame - written_frames
            track += silence_frame * gap
            written_frames += gap
        track += fragment.frames
        written_frames += fragment.nframes

    _write_wav(output_path, ref, bytes(track))


def _write_wav(path: Path, ref: _Fragment, frames: bytes) -> None:
    try:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(ref.nchannels)
            w.setsampwidth(ref.sampwidth)
            w.setframerate(ref.framerate)
            w.writeframes(frames)
    except (wave.Error, OSError) as e:
        raise TtsError(f"unable to write audio file {path}: {e}") from e
