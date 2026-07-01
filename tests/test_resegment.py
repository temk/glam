from glam.resegment import resegment


def test_short_segment_keeps_original_timing():
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there"}]
    cues = resegment(segments, cps=18, max_chars_per_line=42, max_lines=2)

    assert len(cues) == 1
    assert cues[0].start == 0.0
    assert cues[0].end == 3.0
    assert cues[0].text == "hello there"


def test_long_text_wraps_into_multiple_lines():
    text = "this is a longer sentence that should wrap across more than one line of subtitle text"
    segments = [{"start": 0.0, "end": 3.0, "text": text}]
    cues = resegment(segments, cps=18, max_chars_per_line=20, max_lines=2)

    for cue in cues:
        lines = cue.text.split("\n")
        assert len(lines) <= 2
        for line in lines:
            assert len(line) <= 20


def test_text_needing_more_time_extends_into_gap_before_next_segment():
    # 60 chars at cps=18 needs ~3.33s; original window is only 1s
    text = "x" * 60
    segments = [
        {"start": 0.0, "end": 1.0, "text": text},
        {"start": 5.0, "end": 6.0, "text": "next"},
    ]
    cues = resegment(segments, cps=18, max_chars_per_line=42, max_lines=2, min_gap=0.1)

    first = [c for c in cues if c.start == 0.0][0]
    assert first.end > 1.0  # extended past its original end
    assert first.end <= 5.0 - 0.1  # never crosses into the next segment's start


def test_never_overlaps_next_segment_even_when_reading_time_would_need_more():
    text = "x" * 300  # would need ~16.7s at cps=18, way more than available
    segments = [
        {"start": 0.0, "end": 1.0, "text": text},
        {"start": 2.0, "end": 3.0, "text": "next"},
    ]
    cues = resegment(segments, cps=18, max_chars_per_line=42, max_lines=2, min_gap=0.1)

    first_segment_cues = [c for c in cues if c.start < 2.0]
    for cue in first_segment_cues:
        assert cue.end <= 2.0 - 0.1 + 1e-9


def test_last_segment_can_extend_without_a_next_bound():
    text = "x" * 100  # needs ~5.6s at cps=18
    segments = [{"start": 0.0, "end": 1.0, "text": text}]
    cues = resegment(segments, cps=18, max_chars_per_line=42, max_lines=2)

    assert len(cues) >= 1
    assert cues[-1].end >= 5.0


def test_empty_text_segments_are_skipped():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "  "},
        {"start": 1.0, "end": 2.0, "text": "real text"},
    ]
    cues = resegment(segments)
    assert len(cues) == 1
    assert cues[0].text == "real text"


def test_cues_are_chronologically_ordered_and_non_overlapping():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 2.0, "end": 4.0, "text": "b"},
        {"start": 4.0, "end": 6.0, "text": "c"},
    ]
    cues = resegment(segments)
    for prev, cur in zip(cues, cues[1:]):
        assert prev.end <= cur.start + 1e-9
