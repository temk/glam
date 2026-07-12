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

# Force a unit closed once it grows past any of these, even mid-sentence.
MAX_UNIT_DURATION = 12.0
MAX_UNIT_CHARS = 250
MAX_UNIT_SEGMENTS = 5


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
    # Size caps: force the unit closed once it grows too large, even mid-sentence.
    if segment.end - unit.start > MAX_UNIT_DURATION:
        return False
    if len(unit.text) + len(segment.text) > MAX_UNIT_CHARS:
        return False
    if len(unit.source_ids) >= MAX_UNIT_SEGMENTS:
        return False
    # Positive triggers: any one means the sentence is still running.
    if pause < MERGE_MAX_PAUSE and not prev.endswith(SENTENCE_END_CHARS):
        return True
    if _ends_with_continuation(prev):
        return True
    return bool(nxt) and nxt[0].islower()


def _ends_with_continuation(prev: str) -> bool:
    if prev.endswith(CONTINUATION_CHARS):
        return True
    words = prev.split()
    if not words:
        return False
    return words[-1].strip(".,:;!?").lower() in CONTINUATION_WORDS
