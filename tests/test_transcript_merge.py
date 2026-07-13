from glam.transcript_merge import SOFT_MAX_DURATION, merge_sentences
from glam.backend.transcribe.base import AsrSegment


def _seg(id: int, start: float, end: float, text: str) -> AsrSegment:
    return AsrSegment(id=id, start=start, end=end, text=text)


def test_merges_short_pause_when_sentence_unfinished():
    segs = [_seg(0, 0.0, 1.0, "I went to the"), _seg(1, 1.1, 2.0, " store.")]
    out = merge_sentences(segs)

    assert len(out) == 1
    assert out[0].text == "I went to the store."
    assert out[0].start == 0.0 and out[0].end == 2.0
    assert out[0].id == 0
    assert out[0].source_ids == [0, 1]


def test_keeps_finished_sentences_apart():
    segs = [_seg(0, 0.0, 1.0, "Hello."), _seg(1, 1.1, 2.0, "World.")]
    out = merge_sentences(segs)

    assert [s.text for s in out] == ["Hello.", "World."]
    assert [s.source_ids for s in out] == [[0], [1]]


def test_merges_on_continuation_word():
    # pause 1.0 is above the 0.8 merge window, so only the trailing "and" can trigger the merge.
    segs = [_seg(0, 0.0, 1.0, "I like tea and"), _seg(1, 2.0, 3.0, " coffee.")]
    out = merge_sentences(segs)

    assert [s.text for s in out] == ["I like tea and coffee."]


def test_merges_on_trailing_comma():
    segs = [_seg(0, 0.0, 1.0, "First,"), _seg(1, 2.0, 3.0, " second.")]
    out = merge_sentences(segs)

    assert [s.text for s in out] == ["First, second."]


def test_merges_when_next_starts_lowercase_even_after_period():
    # A plain period does not stop a merge when the next fragment clearly continues (lowercase start).
    segs = [_seg(0, 0.0, 1.0, "Something."), _seg(1, 2.0, 3.0, " and more.")]
    out = merge_sentences(segs)

    assert [s.text for s in out] == ["Something. and more."]


def test_long_pause_blocks_merge():
    segs = [_seg(0, 0.0, 1.0, "I was saying"), _seg(1, 3.0, 4.0, " more.")]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0], [1]]


def test_question_mark_blocks_merge():
    segs = [_seg(0, 0.0, 1.0, "Really?"), _seg(1, 1.1, 2.0, " yes.")]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0], [1]]


def test_soft_cap_extends_to_the_next_sentence_boundary():
    # The unit passes the 12s soft cap mid-sentence, keeps going to the period, then closes there.
    segs = [
        _seg(0, 0.0, 6.0, " I was saying"),
        _seg(1, 6.0, 11.0, " a lot of"),
        _seg(2, 11.0, 14.0, " things."),  # period lands past the 12s soft cap
        _seg(3, 14.0, 17.0, " Next point."),
    ]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0, 1, 2], [3]]
    assert out[0].end - out[0].start > SOFT_MAX_DURATION  # extended past the soft cap to the boundary


def test_soft_cap_does_not_cut_mid_sentence():
    # Over the soft duration but never at a boundary and under the hard ceiling: stays one unit.
    segs = [
        _seg(0, 0.0, 7.0, " I keep talking and"),
        _seg(1, 7.0, 13.0, " talking and"),  # now over 12s, but ends on "and"
        _seg(2, 13.0, 15.0, " talking on"),
    ]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0, 1, 2]]
    assert out[0].end - out[0].start > SOFT_MAX_DURATION


def test_absolute_ceiling_cuts_a_boundaryless_run():
    # No sentence boundary at all; the hard duration ceiling forces the cut mid-sentence.
    segs = [
        _seg(0, 0.0, 5.0, " going on and"),
        _seg(1, 5.0, 10.0, " on and"),
        _seg(2, 10.0, 16.0, " on and"),
        _seg(3, 16.0, 23.0, " on forever"),  # would push the unit to 23s, over the 20s ceiling
    ]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0, 1, 2], [3]]
    assert not out[0].text.rstrip().endswith((".", "?", "!"))  # cut mid-sentence by the ceiling


def test_renumbers_and_records_source_ids():
    segs = [_seg(3, 0.0, 1.0, "A."), _seg(7, 1.1, 2.0, "B."), _seg(9, 2.1, 3.0, "C.")]
    out = merge_sentences(segs)

    assert [s.id for s in out] == [0, 1, 2]
    assert [s.source_ids for s in out] == [[3], [7], [9]]


def test_absorbs_tiny_interjection_into_previous():
    # " Oh." (0.32s, 3 chars) can't merge by the normal rules (preceded by "?", followed by uppercase),
    # so it is absorbed into the previous unit rather than left as a stray segment.
    segs = [
        _seg(0, 0.0, 2.0, " Was there a question?"),
        _seg(1, 2.0, 2.32, " Oh."),
        _seg(2, 2.5, 5.0, " Yeah, so next."),
    ]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0, 1], [2]]
    assert out[0].text.strip().endswith("Oh.")


def test_absorbs_tiny_leading_unit_into_next():
    segs = [_seg(0, 0.0, 0.3, " Oh."), _seg(1, 0.5, 3.0, " Yeah, so this is the thing.")]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0, 1]]
    assert out[0].text.strip().startswith("Oh.")


def test_does_not_absorb_short_text_with_long_duration():
    # 3 chars but 1.5s (drawn out): absorbing needs BOTH a short duration and short text.
    segs = [_seg(0, 0.0, 3.0, " Something happened here."), _seg(1, 3.0, 4.5, " Oh.")]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0], [1]]
