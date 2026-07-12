from glam.transcript_merge import merge_sentences
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


def test_force_closes_unit_after_max_segments():
    # Six lowercase fragments all want to merge; the 5-source cap forces a new unit at the sixth.
    words = ["the", " cat", " sat", " on", " the", " mat"]
    segs = [_seg(i, i * 0.2, i * 0.2 + 0.1, w) for i, w in enumerate(words)]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0, 1, 2, 3, 4], [5]]
    assert [s.id for s in out] == [0, 1]


def test_force_closes_unit_after_max_chars():
    segs = [_seg(0, 0.0, 1.0, "a" * 200), _seg(1, 1.1, 2.0, " " + "b" * 100)]
    out = merge_sentences(segs)

    assert [len(s.source_ids) for s in out] == [1, 1]


def test_force_closes_unit_after_max_duration():
    segs = [_seg(0, 0.0, 1.0, "start and"), _seg(1, 1.1, 15.0, " end.")]
    out = merge_sentences(segs)

    assert [s.source_ids for s in out] == [[0], [1]]


def test_renumbers_and_records_source_ids():
    segs = [_seg(3, 0.0, 1.0, "A."), _seg(7, 1.1, 2.0, "B."), _seg(9, 2.1, 3.0, "C.")]
    out = merge_sentences(segs)

    assert [s.id for s in out] == [0, 1, 2]
    assert [s.source_ids for s in out] == [[3], [7], [9]]
