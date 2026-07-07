"""Reading and validating the `translation.json` artifact shared by `subtitles` and `tts`."""

import json
from pathlib import Path
from dataclasses import dataclass

from glam.common.errors import GlamError


class TranslationError(GlamError):
    pass


def translation_filename(target: str) -> str:
    """Per-target name of the `translate` artifact, e.g. `translation.ru.json`."""
    return f"translation.{target}.json"


def fixed_translation_filename(target: str) -> str:
    """Per-target name of the `accent` step's corrected artifact, e.g. `translation.ru.fixed.json`."""
    return f"translation.{target}.fixed.json"


@dataclass
class TranslatedSegment:
    id: int
    start: float
    end: float
    translated_text: str


def load_translated_segments(path: Path) -> list[TranslatedSegment]:
    if not path.exists():
        raise TranslationError(f"missing translation artifact: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise TranslationError(f"invalid translation {path}: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise TranslationError(f"invalid translation {path}: missing 'segments' list")
    return [_segment(raw, path) for raw in data["segments"]]


def _segment(raw: object, path: Path) -> TranslatedSegment:
    if not isinstance(raw, dict):
        raise TranslationError(f"invalid translation {path}: segment is not an object")
    for key in ("id", "start", "end", "translated_text"):
        if raw.get(key) is None:
            raise TranslationError(f"invalid translation {path}: segment {raw.get('id')} is missing '{key}'")
    start, end = raw["start"], raw["end"]
    for label, value in (("start", start), ("end", end)):
        # bool is an int subclass; reject it so a stray true/false is not read as a timestamp.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TranslationError(f"invalid translation {path}: segment {raw['id']} has non-numeric {label}")
    if end <= start:
        raise TranslationError(f"invalid translation {path}: segment {raw['id']} has end <= start ({end} <= {start})")
    text = raw["translated_text"]
    if not isinstance(text, str):
        raise TranslationError(f"invalid translation {path}: segment {raw['id']} has non-string translated_text")
    if not text.strip():
        raise TranslationError(f"invalid translation {path}: segment {raw['id']} has empty translated_text")
    return TranslatedSegment(id=int(raw["id"]), start=float(start), end=float(end), translated_text=text)
