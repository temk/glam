import re
import json
from pathlib import Path
from collections.abc import Callable

from glam.common.job import JOB_MANIFEST_NAME, read_job_manifest
from glam.common.config import Config
from glam.common.errors import GlamError
from glam.common.translation import translation_filename, fixed_translation_filename

# Combining acute accent placed on the stressed vowel; this is the stress form TTS engines expect.
STRESS_MARK = "́"

# Silero marks the stressed vowel by writing `+` in front of it (`зов+ут`); this matches such a `+`
# and the vowel that follows so we can move the mark onto the vowel as a combining accent.
_STRESS_PLUS = re.compile(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])")


class AccentError(GlamError):
    pass


def run(job_id: str, config: Config, target: str | None = None, force: bool = False, echo=print) -> Path | None:
    """Apply the target language's text fixer to `translation.<target>.json` -> `translation.<target>.fixed.json`.

    Only languages with a registered fixer produce output; for any other language the step is a no-op.
    """
    job_path = config.job_dir / job_id
    if not job_path.is_dir():
        raise AccentError(f"job not found: {job_id} (looked in {job_path})")

    manifest = read_job_manifest(job_path / JOB_MANIFEST_NAME)
    target = target or manifest.languages.target
    if not target:
        raise AccentError("missing target language: pass --target or set languages.target in job.yaml")

    fixer = _FIXERS.get(target)
    if fixer is None:
        echo(f"no text fixer for target '{target}'; nothing to do")
        return None

    fixed_path = job_path / fixed_translation_filename(target)
    if fixed_path.exists() and not force:
        echo(f"skip accent, already exists: {fixed_path}")
        return fixed_path

    doc = _load_translation_doc(job_path / translation_filename(target))
    segments = doc["segments"]
    fixed = fixer([str(seg.get("translated_text") or "") for seg in segments])
    for seg, text in zip(segments, fixed):
        if seg.get("translated_text"):
            seg["translated_text"] = text

    fixed_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    echo(f"wrote {fixed_path} ({len(segments)} segments)")
    return fixed_path


def _load_translation_doc(path: Path) -> dict:
    if not path.exists():
        raise AccentError(f"missing translation artifact: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise AccentError(f"invalid translation {path}: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise AccentError(f"invalid translation {path}: missing 'segments' list")
    return data


def _fix_russian_stress(texts: list[str]) -> list[str]:
    """Mark the stressed vowel in each Russian segment with a combining acute accent."""
    accentor = _load_ru_accentor()
    # Silero takes one sentence at a time and emits `+` before each stressed vowel; move the mark
    # onto the vowel as a combining acute accent (`зов+ут` -> `зову́т`) that TTS engines understand.
    # `stress_single_vowel=False` leaves monosyllables unmarked — their stress is unambiguous and
    # marking every one just adds noise the TTS voice does not need.
    return [_plus_to_combining(accentor(t, stress_single_vowel=False)) if t.strip() else t for t in texts]


def _plus_to_combining(text: str) -> str:
    return _STRESS_PLUS.sub(r"\1" + STRESS_MARK, text)


def _load_ru_accentor():
    # Imported lazily: this pulls in torch and the accentor model, which no other step needs.
    from silero_stress import load_accentor  # type: ignore[import-untyped]

    return load_accentor()


# Target language -> text fixer. Languages without an entry are left untouched.
_FIXERS: dict[str, Callable[[list[str]], list[str]]] = {
    "ru": _fix_russian_stress,
}
