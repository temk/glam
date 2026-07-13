import json
from dacite import DaciteError
from dacite import from_dict as dacite_from_dict
from pathlib import Path
from dataclasses import asdict, dataclass

from glam.common.job import JOB_MANIFEST_NAME, JobManifest, read_job_manifest
from glam.common.hooks import service_hooks
from glam.common.config import Config, ServiceName
from glam.common.errors import GlamError
from glam.transcript_merge import merge_sentences
from glam.transcript_cleanup import clean_segments
from glam.backend.transcribe.base import AsrSegment, build_transcribe_backend

TRANSCRIPT_NAME = "transcript.json"
TRANSCRIPT_RAW_NAME = "transcript.raw.json"
CLEANUP_NAME = "transcript.cleanup.json"
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


def run(job_id: str, config: Config, force: bool = False, strict: bool = False, echo=print) -> Path:
    """Transcribe a job's audio through the configured ASR service into `transcript.json`.

    Both artifacts are produced lazily: `transcript.raw.json` is regenerated (a fresh ASR call) only
    when it is missing, and `transcript.json` is re-cleaned from the raw transcript only when it is
    missing. `--force` regenerates both. So deleting `transcript.json` alone re-cleans from the cached
    raw transcript without a new ASR call.
    """
    job_path = config.job_dir / job_id
    if not job_path.is_dir():
        raise TranscribeError(f"job not found: {job_id} (looked in {job_path})")

    manifest = read_job_manifest(job_path / JOB_MANIFEST_NAME)

    transcript_path = job_path / TRANSCRIPT_NAME
    if transcript_path.exists() and not force:
        echo(f"skip transcript, already exists: {transcript_path}")
        return transcript_path

    # Look up the service's hooks without requiring the service itself: re-cleaning from a cached
    # transcript.raw.json needs no ASR service, and `_obtain_raw` still raises if it is absent when ASR
    # is actually needed.
    service = next((s for s in config.services if s.name == ServiceName.TRANSCRIBE), None)
    with service_hooks(service.hooks if service is not None else None, echo):
        raw = _obtain_raw(job_id, job_path, manifest, config, force, echo)

        # transcript.json is healed in two passes: clean the raw segments, then merge the survivors into
        # sentence-level units (each keeps its source ids). Serialize the raw's top-level fields, then swap
        # in the merged segments, whose `source_ids` field the raw AsrSegment schema does not carry.
        result = clean_segments(raw.segments, strict=strict)
        merged = merge_sentences(result.segments)
        document = asdict(raw)
        document["segments"] = [asdict(segment) for segment in merged]
        transcript_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")

        cleanup = {
            "version": TRANSCRIPT_VERSION,
            "step": "transcribe",
            "job_id": job_id,
            "warnings": [asdict(w) for w in result.warnings],
        }
        (job_path / CLEANUP_NAME).write_text(json.dumps(cleanup, ensure_ascii=False, indent=2) + "\n")

    echo(
        f"wrote {transcript_path} ({len(merged)} segments from {len(result.segments)} cleaned, "
        f"{len(result.warnings)} warnings)"
    )
    return transcript_path


def _obtain_raw(job_id: str, job_path: Path, manifest: JobManifest, config: Config, force: bool, echo) -> Transcript:
    """Reuse a cached `transcript.raw.json` when present; otherwise call the ASR service and write it."""
    raw_path = job_path / TRANSCRIPT_RAW_NAME
    if raw_path.exists() and not force:
        echo(f"reuse raw transcript: {raw_path}")
        return _load_raw(raw_path)

    audio_path = job_path / manifest.source.audio_artifact
    if not audio_path.exists():
        raise TranscribeError(f"missing audio artifact: {audio_path}")

    backend = build_transcribe_backend(config[ServiceName.TRANSCRIBE])
    segments = backend.transcribe(audio_path, manifest.languages.source)
    raw = Transcript(
        version=TRANSCRIPT_VERSION,
        step="transcribe",
        job_id=job_id,
        source_language=manifest.languages.source,
        model=backend.model,
        audio_artifact=manifest.source.audio_artifact,
        segments=segments,
    )
    raw_path.write_text(json.dumps(asdict(raw), ensure_ascii=False, indent=2) + "\n")
    echo(f"wrote {raw_path} ({len(segments)} segments)")
    return raw


def _load_raw(path: Path) -> Transcript:
    try:
        data = json.loads(path.read_text())
        return dacite_from_dict(data_class=Transcript, data=data)
    except (json.JSONDecodeError, DaciteError) as e:
        raise TranscribeError(f"invalid raw transcript {path}: {e}") from e
