"""Job manifest (`job.yaml`): the shared source of truth about a single job."""

import yaml
from dacite import DaciteError
from dacite import from_dict as dacite_from_dict
from pathlib import Path
from dataclasses import asdict, dataclass

from glam.common.errors import GlamError

JOB_MANIFEST_NAME = "job.yaml"
JOB_MANIFEST_VERSION = 1


class JobError(GlamError):
    """Raised when a job manifest (`job.yaml`) is missing or malformed."""


@dataclass
class JobInfo:
    id: str
    created_at: str


@dataclass
class SourceInfo:
    original_path: str
    filename: str
    artifact: str
    audio_artifact: str
    duration_seconds: float


@dataclass
class Languages:
    source: str
    target: str


@dataclass
class JobManifest:
    """In-memory representation of a job's `job.yaml` manifest created by `init`.

    Mirrors the manifest schema in docs/architecture.md and docs/steps/init.md and
    is the shared source of truth about a job for every downstream step.
    """

    version: int
    job: JobInfo
    source: SourceInfo
    languages: Languages
    voice: str | None = None


def read_job_manifest(path: str | Path) -> JobManifest:
    try:
        data = yaml.safe_load(Path(path).read_text())
        return dacite_from_dict(data_class=JobManifest, data=data)
    except (OSError, DaciteError, yaml.YAMLError) as e:
        raise JobError(f"invalid job manifest {path}: {e}") from e


def write_job_manifest(manifest: JobManifest, path: str | Path) -> None:
    with Path(path).open("w") as f:
        yaml.safe_dump(asdict(manifest), f, sort_keys=False, allow_unicode=True)
