from glam.transcript_cleanup import MAX_CHARS_PER_SECOND, clean_segments
from glam.backend.transcribe.base import AsrSegment


def _seg(id: int, start: float, end: float, text: str) -> AsrSegment:
    return AsrSegment(id=id, start=start, end=end, text=text)


def _rules(result) -> list[str]:
    return [w.rule for w in result.warnings]


def test_clean_input_produces_no_warnings():
    segments = [_seg(0, 0.0, 1.5, "Hello."), _seg(1, 1.5, 3.0, "World.")]
    result = clean_segments(segments)

    assert [s.text for s in result.segments] == ["Hello.", "World."]
    assert result.warnings == []


def test_drops_punctuation_only_segment():
    segments = [_seg(0, 0.0, 1.0, "Hello."), _seg(1, 1.0, 1.2, "..."), _seg(2, 1.2, 2.0, "World.")]
    result = clean_segments(segments)

    assert [s.text for s in result.segments] == ["Hello.", "World."]
    warning = next(w for w in result.warnings if w.rule == "punctuation_only")
    assert warning.segment_id == 1
    assert warning.text == "..."


def test_bad_timing_is_warned_but_kept():
    segments = [_seg(0, 2.0, 2.0, "Zero length."), _seg(1, 5.0, 4.0, "Reversed.")]
    result = clean_segments(segments)

    assert [s.text for s in result.segments] == ["Zero length.", "Reversed."]
    assert [w.segment_id for w in result.warnings if w.rule == "bad_timing"] == [0, 1]


def test_fast_speech_is_warned_but_kept():
    text = "x" * 100  # 100 chars in 1s = 100 cps, far above the limit
    assert 100 > MAX_CHARS_PER_SECOND
    segments = [_seg(0, 0.0, 1.0, text)]
    result = clean_segments(segments)

    assert len(result.segments) == 1
    assert _rules(result) == ["fast_speech"]
    assert result.warnings[0].segment_id == 0


def test_collapses_consecutive_identical_segments():
    segments = [_seg(0, 0.0, 1.0, "Repeat."), _seg(1, 1.0, 2.0, "Repeat."), _seg(2, 2.0, 3.0, "Next.")]
    result = clean_segments(segments)

    assert [s.text for s in result.segments] == ["Repeat.", "Next."]
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 2.0  # end extended over the merged duplicate
    duplicate = next(w for w in result.warnings if w.rule == "duplicate_segment")
    assert duplicate.segment_id == 1


def test_collapses_repeat_then_continuation():
    segments = [_seg(0, 0.0, 2.0, "The quick"), _seg(1, 2.0, 4.0, "The quick brown fox")]
    result = clean_segments(segments)

    assert [s.text for s in result.segments] == ["The quick brown fox"]
    assert result.segments[0].start == 0.0  # survivor's start pulled back over the dropped segment
    assert result.segments[0].end == 4.0
    assert _rules(result) == ["repeat_continuation"]
    assert result.warnings[0].segment_id == 0


def test_continuation_ignores_mid_word_prefix():
    segments = [_seg(0, 0.0, 1.0, "Hell"), _seg(1, 1.0, 2.0, "Hello there")]
    result = clean_segments(segments)

    assert [s.text for s in result.segments] == ["Hell", "Hello there"]
    assert _rules(result) == []


def test_renumbers_ids_sequentially():
    segments = [_seg(0, 0.0, 1.0, "One."), _seg(5, 1.0, 1.1, "."), _seg(9, 1.1, 2.0, "Three.")]
    result = clean_segments(segments)

    assert [s.id for s in result.segments] == [0, 1]


def test_fillers_are_kept_without_strict():
    segments = [_seg(0, 0.0, 1.0, "э.."), _seg(1, 1.0, 3.0, "Так, э, давайте начнём.")]
    result = clean_segments(segments)

    assert [s.text for s in result.segments] == ["э..", "Так, э, давайте начнём."]
    assert _rules(result) == []


def test_strict_drops_filler_only_segment():
    segments = [_seg(0, 0.0, 1.0, "Hello."), _seg(1, 1.0, 1.3, "э.."), _seg(2, 1.3, 2.0, "World.")]
    result = clean_segments(segments, strict=True)

    assert [s.text for s in result.segments] == ["Hello.", "World."]
    warning = next(w for w in result.warnings if w.rule == "filler_only")
    assert warning.segment_id == 1
    assert warning.text == "э.."


def test_strict_drops_english_and_hummed_fillers():
    segments = [_seg(0, 0.0, 1.0, "Uh"), _seg(1, 1.0, 2.0, "Hmm,"), _seg(2, 2.0, 3.0, "Okay.")]
    result = clean_segments(segments, strict=True)

    assert [s.text for s in result.segments] == ["Okay."]
    assert _rules(result) == ["filler_only", "filler_only"]


def test_strict_strips_embedded_fillers_and_keeps_timestamps():
    segments = [_seg(0, 0.0, 3.0, "Так, э, давайте начнём.")]
    result = clean_segments(segments, strict=True)

    assert [s.text for s in result.segments] == ["Так, давайте начнём."]
    assert result.segments[0].start == 0.0 and result.segments[0].end == 3.0
    assert _rules(result) == ["filler_removed"]


def test_strict_preserves_leading_space_when_stripping_fillers():
    # Whisper fragments start with a space; the merge step relies on it, so it must survive.
    segments = [_seg(0, 0.0, 3.0, " um, we should start")]
    result = clean_segments(segments, strict=True)

    assert result.segments[0].text == " we should start"


def test_strict_keeps_real_words_that_contain_a_filler():
    # "Ummm" is a filler, but "summer"/"этаж" are real words that merely contain those letters.
    segments = [_seg(0, 0.0, 2.0, "It was summer."), _seg(1, 2.0, 4.0, "Верхний этаж.")]
    result = clean_segments(segments, strict=True)

    assert [s.text for s in result.segments] == ["It was summer.", "Верхний этаж."]
    assert _rules(result) == []


def test_combined_pipeline_and_warning_ids_reference_raw_ids():
    segments = [
        _seg(0, 0.0, 1.0, "Intro"),
        _seg(1, 1.0, 1.1, "."),
        _seg(2, 1.1, 2.0, "Body."),
        _seg(3, 2.0, 3.0, "Body."),
        _seg(4, 3.0, 4.0, "Body. And more."),
    ]
    result = clean_segments(segments)

    assert [s.text for s in result.segments] == ["Intro", "Body. And more."]
    assert [s.id for s in result.segments] == [0, 1]
    assert _rules(result) == ["punctuation_only", "duplicate_segment", "repeat_continuation"]
    assert [w.segment_id for w in result.warnings] == [1, 3, 2]
