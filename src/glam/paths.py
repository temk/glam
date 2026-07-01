import re
from pathlib import Path

from glam.errors import GlamError


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "job"


def job_dir(jobs_root: Path, video_id: str) -> Path:
    return jobs_root / video_id


def resolve_artifact(job_path, glob_pattern, explicit_path=None, exact_candidate=None, hint=""):
    """Locate a single upstream job-dir file, the shared pattern every step uses to consume
    another step's output without calling it directly (steps stay file-based, per architecture.md).

    Precedence: explicit_path (user override) > exact_candidate (derived from a sibling
    step's configured model) > the sole file matching glob_pattern. Raises GlamError if the
    glob is empty or ambiguous.
    """
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise GlamError(f"{path} not found")
        return path

    if exact_candidate is not None and Path(exact_candidate).exists():
        return Path(exact_candidate)

    candidates = sorted(job_path.glob(glob_pattern))
    if not candidates:
        suffix = f" — {hint}" if hint else ""
        raise GlamError(f"no file matching '{glob_pattern}' found in {job_path}{suffix}")
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise GlamError(
            f"multiple files matching '{glob_pattern}' found in {job_path} ({names}) "
            "— pass an explicit path to disambiguate"
        )
    return candidates[0]
