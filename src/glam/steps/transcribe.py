import json
from pathlib import Path
from dataclasses import asdict, dataclass

from glam.common.job import JOB_MANIFEST_NAME, read_job_manifest
from glam.common.config import Config, ServiceName
from glam.common.errors import GlamError
from glam.backend.transcribe.base import AsrSegment, build_transcribe_backend

TRANSCRIPT_NAME = "transcript.json"
TRANSCRIPT_VERSION = 1


class TranscribeError(GlamError):
    pass


@dataclass
class Transcript:
    version: int
    step: str
    job_id: str
    source_language: str
    model: str
    audio_artifact: str
    segments: list[AsrSegment]


def run(job_id: str, config: Config, force: bool = False, echo=print) -> Path:
    """Transcribe a job's audio through the configured ASR service into `transcript.json`."""
    job_path = config.job_dir / job_id
    if not job_path.is_dir():
        raise TranscribeError(f"job not found: {job_id} (looked in {job_path})")

    manifest = read_job_manifest(job_path / JOB_MANIFEST_NAME)
    audio_path = job_path / manifest.source.audio_artifact
    if not audio_path.exists():
        raise TranscribeError(f"missing audio artifact: {audio_path}")

    transcript_path = job_path / TRANSCRIPT_NAME
    if transcript_path.exists() and not force:
        echo(f"skip transcript, already exists: {transcript_path}")
        return transcript_path

    backend = build_transcribe_backend(config[ServiceName.TRANSCRIBE])
    segments = backend.transcribe(audio_path, manifest.languages.source)

    transcript = Transcript(
        version=TRANSCRIPT_VERSION,
        step="transcribe",
        job_id=job_id,
        source_language=manifest.languages.source,
        model=backend.model,
        audio_artifact=manifest.source.audio_artifact,
        segments=segments,
    )
    transcript_path.write_text(json.dumps(asdict(transcript), ensure_ascii=False, indent=2) + "\n")
    echo(f"wrote {transcript_path} ({len(segments)} segments)")
    return transcript_path
