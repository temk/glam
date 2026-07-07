import textwrap
from dataclasses import dataclass

DEFAULT_CPS = 18
DEFAULT_MAX_CHARS_PER_LINE = 42
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_GAP = 0.08
MIN_CUE_DURATION = 0.5


@dataclass
class Cue:
    start: float
    end: float
    text: str


def resegment(
    segments,
    cps=DEFAULT_CPS,
    max_chars_per_line=DEFAULT_MAX_CHARS_PER_LINE,
    max_lines=DEFAULT_MAX_LINES,
    min_gap=DEFAULT_MIN_GAP,
):
    """Turn translated segments into subtitle cues re-timed for reading speed.

    Russian translations commonly run longer than the English source, so a segment's
    original [start, end] window is a floor, not a fixed size: if the text needs more
    time to read than the window gives it, the window is extended — capped by the next
    segment's start (minus min_gap) so cues never overlap. If even that isn't enough,
    cues are compressed below the ideal reading duration rather than pushing into the
    next segment; there's no cross-video re-timing cascade in this version.
    """
    cues = []
    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg["start"]
        end = seg["end"]

        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else None
        max_end = (next_start - min_gap) if next_start is not None else None

        chunks = _split_into_chunks(text, max_lines, max_chars_per_line)
        durations = [max(len(c) / cps, MIN_CUE_DURATION) for c in chunks]
        total_needed = sum(durations)

        span = max(end - start, total_needed)
        if max_end is not None:
            span = min(span, max(max_end - start, MIN_CUE_DURATION))

        cue_start = start
        for chunk, dur in zip(chunks, durations):
            share = span * (dur / total_needed) if total_needed > 0 else span / len(chunks)
            cue_end = cue_start + share
            wrapped = "\n".join(_wrap_lines(chunk, max_chars_per_line)[:max_lines])
            cues.append(Cue(start=cue_start, end=cue_end, text=wrapped))
            cue_start = cue_end
    return cues


def _wrap_lines(text, max_chars_per_line):
    return textwrap.wrap(text, width=max_chars_per_line, break_long_words=False, break_on_hyphens=False) or [""]


def _split_into_chunks(text, max_lines, max_chars_per_line):
    """Split text into pieces that each fit within max_lines lines of max_chars_per_line."""
    max_chars = max_lines * max_chars_per_line
    words = text.split()
    if not words:
        return [text]

    pieces = []
    current = []
    current_len = 0
    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > max_chars:
            pieces.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len
    if current:
        pieces.append(" ".join(current))
    return pieces
