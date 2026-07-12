from dataclasses import replace, dataclass

from glam.backend.transcribe.base import AsrSegment

# Above this character rate a segment's timing is almost certainly an ASR glitch rather than real
# speech (~25 chars/sec is roughly twice a fast English speaker). Warn-only; the segment is kept.
MAX_CHARS_PER_SECOND = 25.0


@dataclass
class CleanupWarning:
    rule: str
    segment_id: int  # id in the raw transcript (before renumbering)
    text: str
    detail: str


@dataclass
class CleanupResult:
    segments: list[AsrSegment]
    warnings: list[CleanupWarning]


def clean_segments(segments: list[AsrSegment]) -> CleanupResult:
    """Heal raw ASR segments and collect warnings; warnings reference the raw (pre-renumber) ids."""
    warnings: list[CleanupWarning] = []
    segs = list(segments)

    segs, dropped = _drop_punctuation_only(segs)
    warnings.extend(dropped)
    warnings.extend(_check_timing(segs))  # detection only
    warnings.extend(_check_speed(segs))  # detection only
    segs, merged = _collapse_duplicates(segs)
    warnings.extend(merged)
    segs, continued = _collapse_continuation(segs)
    warnings.extend(continued)
    segs = _renumber(segs)

    return CleanupResult(segments=segs, warnings=warnings)


def _is_punctuation_only(text: str) -> bool:
    return not any(ch.isalnum() for ch in text)


def _drop_punctuation_only(segments: list[AsrSegment]) -> tuple[list[AsrSegment], list[CleanupWarning]]:
    kept: list[AsrSegment] = []
    warnings: list[CleanupWarning] = []
    for seg in segments:
        if _is_punctuation_only(seg.text):
            warnings.append(CleanupWarning("punctuation_only", seg.id, seg.text, "removed punctuation-only segment"))
        else:
            kept.append(seg)
    return kept, warnings


def _check_timing(segments: list[AsrSegment]) -> list[CleanupWarning]:
    warnings: list[CleanupWarning] = []
    for seg in segments:
        if seg.end <= seg.start:
            detail = f"end ({seg.end}) is not after start ({seg.start})"
            warnings.append(CleanupWarning("bad_timing", seg.id, seg.text, detail))
    return warnings


def _check_speed(segments: list[AsrSegment]) -> list[CleanupWarning]:
    warnings: list[CleanupWarning] = []
    for seg in segments:
        duration = seg.end - seg.start
        if duration <= 0:
            continue  # invalid timing is already reported by _check_timing
        cps = len(seg.text.strip()) / duration
        if cps > MAX_CHARS_PER_SECOND:
            detail = f"{cps:.1f} chars/sec exceeds the {MAX_CHARS_PER_SECOND:g} chars/sec limit"
            warnings.append(CleanupWarning("fast_speech", seg.id, seg.text, detail))
    return warnings


def _collapse_duplicates(segments: list[AsrSegment]) -> tuple[list[AsrSegment], list[CleanupWarning]]:
    kept: list[AsrSegment] = []
    warnings: list[CleanupWarning] = []
    for seg in segments:
        if kept and kept[-1].text.strip() == seg.text.strip():
            kept[-1] = replace(kept[-1], end=max(kept[-1].end, seg.end))
            warnings.append(
                CleanupWarning("duplicate_segment", seg.id, seg.text, "merged into previous identical segment")
            )
        else:
            kept.append(seg)
    return kept, warnings


def _is_continuation(prev_text: str, next_text: str) -> bool:
    """True when `next_text` repeats `prev_text` as a whole-word prefix and adds more (a Whisper tail artifact)."""
    if not prev_text or len(next_text) <= len(prev_text) or not next_text.startswith(prev_text):
        return False
    return not next_text[len(prev_text)].isalnum()  # boundary must not fall inside a word


def _collapse_continuation(segments: list[AsrSegment]) -> tuple[list[AsrSegment], list[CleanupWarning]]:
    pending = list(segments)
    kept: list[AsrSegment] = []
    warnings: list[CleanupWarning] = []
    index = 0
    while index < len(pending):
        current = pending[index]
        if index + 1 < len(pending) and _is_continuation(current.text.strip(), pending[index + 1].text.strip()):
            follower = pending[index + 1]
            pending[index + 1] = replace(follower, start=current.start)
            warnings.append(
                CleanupWarning(
                    "repeat_continuation", current.id, current.text, "next segment repeats this text and continues it"
                )
            )
            index += 1
            continue
        kept.append(current)
        index += 1
    return kept, warnings


def _renumber(segments: list[AsrSegment]) -> list[AsrSegment]:
    return [replace(seg, id=index) for index, seg in enumerate(segments)]
