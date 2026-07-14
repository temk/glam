import re
import json
from pathlib import Path
from datetime import datetime

from glam import media
from glam.common.job import (
    JOB_MANIFEST_NAME,
    JOB_MANIFEST_VERSION,
    JobInfo,
    Languages,
    SourceInfo,
    JobManifest,
    write_job_manifest,
)
from glam.common.errors import GlamError


class GlossaryError(GlamError):
    pass


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "job"


def run(
    video_file: str | Path,
    source_lang: str,
    target_lang: str,
    glossary_path: str | Path | None = None,
    voice: str | None = None,
    job_id: str | None = None,
    jobs_root: Path = Path("jobs"),
    force: bool = False,
    echo=print,
) -> Path:
    """Register a job: create its directory, source/audio/glossary artifacts, and the job.yaml manifest."""
    video_file = Path(video_file).resolve()
    if job_id is None:
        job_id = slugify(video_file.stem)
        echo(f"generated job id: {job_id}")

    job_path = jobs_root / job_id
    job_path.mkdir(parents=True, exist_ok=True)

    source_artifact = f"source{video_file.suffix}"
    source_link = job_path / source_artifact
    if _needs_build(source_link, force, "source link", echo):
        source_link.symlink_to(video_file)
        echo(f"linked {source_link} -> {video_file}")

    audio_path = job_path / "audio.wav"
    if _needs_build(audio_path, force, "audio", echo):
        media.extract_audio(video_file, audio_path)
        echo(f"wrote {audio_path}")

    glossary_dst = job_path / "glossary.json"
    if _needs_build(glossary_dst, force, "glossary", echo):
        _write_glossary(glossary_dst, glossary_path)
        echo(f"wrote {glossary_dst}")

    manifest_path = job_path / JOB_MANIFEST_NAME
    if _needs_build(manifest_path, force, "manifest", echo):
        manifest = _build_manifest(
            job_id, video_file, source_artifact, audio_path.name, source_lang, target_lang, voice
        )
        write_job_manifest(manifest, manifest_path)
        echo(f"wrote {manifest_path}")

    return job_path


def _needs_build(path: Path, force: bool, label: str, echo) -> bool:
    """Decide whether to (re)create `path`. Skip and log if it exists without `--force`;
    with `--force`, remove the stale artifact so the caller can rebuild it."""
    if path.exists() and not force:
        echo(f"skip {label}, already exists: {path}")
        return False
    if path.is_symlink() or path.exists():
        path.unlink()
    return True


def _write_glossary(dst: Path, glossary_path: str | Path | None) -> None:
    """Write the job-local `glossary.json` as a JSON string→string map (see docs/steps/init.md)."""
    mapping = {} if glossary_path is None else _load_glossary_map(glossary_path)
    dst.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")


def _load_glossary_map(glossary_path: str | Path) -> dict[str, str]:
    """Normalize a glossary input into a term→translation map.

    A `.json` file is parsed by content: an object is taken as-is, an array becomes an
    identity map (`term -> term`). Any other file is read as text, one term per line.
    """
    path = Path(glossary_path)
    if not path.is_file():
        raise GlossaryError(f"glossary file not found: {path}")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise GlossaryError(f"invalid glossary JSON {path}: {e}") from e
        if isinstance(data, dict):
            return _validate_map(data, path)
        if isinstance(data, list):
            return _terms_to_map(data, path)
        raise GlossaryError(f"glossary {path} must contain a JSON object or array, got {type(data).__name__}")

    lines = (line.strip() for line in path.read_text().splitlines())
    terms = [line for line in lines if line and not line.startswith("#")]
    return {term: term for term in terms}


def _terms_to_map(terms: list, path: Path) -> dict[str, str]:
    if not all(isinstance(term, str) for term in terms):
        raise GlossaryError(f"glossary array in {path} must contain only strings")
    return {term: term for term in terms}


def _validate_map(data: dict, path: Path) -> dict[str, str]:
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise GlossaryError(f"glossary object in {path} must map strings to strings")
    return data


def _build_manifest(
    job_id: str,
    video_file: Path,
    source_artifact: str,
    audio_artifact: str,
    source_lang: str,
    target_lang: str,
    voice: str | None,
) -> JobManifest:
    return JobManifest(
        version=JOB_MANIFEST_VERSION,
        job=JobInfo(id=job_id, created_at=datetime.now().astimezone().isoformat()),
        source=SourceInfo(
            original_path=str(video_file),
            filename=video_file.name,
            artifact=source_artifact,
            audio_artifact=audio_artifact,
            duration_seconds=media.probe_duration(video_file),
        ),
        languages=Languages(source=source_lang, target=target_lang),
        voice=voice,
    )
