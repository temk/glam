from dataclasses import dataclass

from glam.backend.transcribe.base import AsrSegment

# Tokens/characters that mark the previous segment as grammatically unfinished, so the next segment
# continues the same sentence and should be merged into it.
CONTINUATION_WORDS = frozenset({"and", "or", "but", "because", "which", "that", "so", "uh", "um"})
CONTINUATION_CHARS = (",", ":")
SENTENCE_END_CHARS = (".", "?", "!")
# A finished question/exclamation ends the sentence outright (a plain "." does not, since ASR
# sprinkles periods mid-thought).
HARD_STOP_CHARS = ("?", "!")

# Merge when the gap is this short (and the sentence is unfinished); never merge across a gap this long.
MERGE_MAX_PAUSE = 0.8
NO_MERGE_PAUSE = 1.2

# Soft target: once a unit reaches one of these it is closed — but only at a clean sentence boundary
# (it ends on `.?!`). Mid-sentence the unit keeps growing toward the next boundary.
SOFT_MAX_DURATION = 12.0
SOFT_MAX_CHARS = 250
SOFT_MAX_SEGMENTS = 5

# Absolute ceiling: stop growing even mid-sentence, so a boundary-less run cannot grow without bound.
HARD_MAX_DURATION = 20.0
HARD_MAX_CHARS = 400
HARD_MAX_SEGMENTS = 8


@dataclass
class MergedSegment:
    id: int
    start: float
    end: float
    text: str
    source_ids: list[int]


@dataclass
class _Unit:
    start: float
    end: float
    text: str
    source_ids: list[int]


def merge_sentences(segments: list[AsrSegment]) -> list[MergedSegment]:
    """Merge adjacent ASR fragments into sentence-level units so a sentence is not split across ids.

    Each unit keeps `source_ids`, the raw ids it absorbed, for debugging. Units are renumbered from 0.
    """
    units: list[_Unit] = []
    for segment in segments:
        if units and _should_merge(units[-1], segment):
            unit = units[-1]
            unit.text += segment.text
            unit.end = segment.end
            unit.source_ids.append(segment.id)
        else:
            units.append(_Unit(segment.start, segment.end, segment.text, [segment.id]))
    return [MergedSegment(index, u.start, u.end, u.text, u.source_ids) for index, u in enumerate(units)]


def _should_merge(unit: _Unit, segment: AsrSegment) -> bool:
    prev = unit.text.rstrip()
    nxt = segment.text.strip()
    pause = segment.start - unit.end

    # Hard vetoes: a long gap or a finished question/exclamation ends the sentence.
    if pause > NO_MERGE_PAUSE or prev.endswith(HARD_STOP_CHARS):
        return False
    # Absolute ceiling: stop even mid-sentence so a boundary-less run stays bounded.
    if _merge_exceeds(unit, segment, HARD_MAX_DURATION, HARD_MAX_CHARS, HARD_MAX_SEGMENTS):
        return False
    # Soft target: a large unit is closed only at a clean sentence boundary; otherwise it keeps
    # merging toward the next boundary (bounded by the absolute ceiling above).
    if prev.endswith(SENTENCE_END_CHARS) and _unit_reached(unit, SOFT_MAX_DURATION, SOFT_MAX_CHARS, SOFT_MAX_SEGMENTS):
        return False
    # Positive triggers: any one means the sentence is still running.
    if pause < MERGE_MAX_PAUSE and not prev.endswith(SENTENCE_END_CHARS):
        return True
    if _ends_with_continuation(prev):
        return True
    return bool(nxt) and nxt[0].islower()


def _unit_reached(unit: _Unit, max_duration: float, max_chars: int, max_segments: int) -> bool:
    return unit.end - unit.start >= max_duration or len(unit.text) >= max_chars or len(unit.source_ids) >= max_segments


def _merge_exceeds(unit: _Unit, segment: AsrSegment, max_duration: float, max_chars: int, max_segments: int) -> bool:
    return (
        segment.end - unit.start > max_duration
        or len(unit.text) + len(segment.text) > max_chars
        or len(unit.source_ids) >= max_segments
    )


def _ends_with_continuation(prev: str) -> bool:
    if prev.endswith(CONTINUATION_CHARS):
        return True
    words = prev.split()
    if not words:
        return False
    return words[-1].strip(".,:;!?").lower() in CONTINUATION_WORDS
