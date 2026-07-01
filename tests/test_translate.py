import json

import pytest

from glam.errors import GlamError
from glam.glossary import GlossaryError, load_glossary
from glam.steps import translate


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeChatCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeChatCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        return FakeChatCompletion(content)


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeChatCompletions(responses)


class FakeClient:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


def make_config(model="fake-model", glossary=None, **overrides):
    translate_cfg = {
        "backend": "openai_compatible",
        "base_url": "http://fake-host:11434/v1",
        "model": model,
    }
    if glossary:
        translate_cfg["glossary"] = str(glossary)
    translate_cfg.update(overrides)
    return {"steps": {"translate": translate_cfg}}


def write_transcript(job_dir, asr_model="fake-asr", texts=None):
    job_dir.mkdir(parents=True, exist_ok=True)
    texts = texts or ["hello", "world"]
    segments = [
        {"id": i, "start": float(i), "end": float(i + 1), "text": t}
        for i, t in enumerate(texts)
    ]
    from glam.paths import slugify
    path = job_dir / f"transcript.{slugify(asr_model)}.json"
    path.write_text(json.dumps({"model": asr_model, "segments": segments}))
    return path


def test_translate_writes_segments(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_transcript(job_dir)

    fake_client = FakeClient([json.dumps({"translations": ["привет", "мир"]})])
    monkeypatch.setattr(translate, "build_openai_client", lambda cfg: fake_client)

    output_path = translate.run(
        "myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    assert output_path == job_dir / "translation.ru.fake-model.json"
    data = json.loads(output_path.read_text())
    assert data["lang"] == "ru"
    assert [s["text"] for s in data["segments"]] == ["привет", "мир"]
    assert [s["text_source"] for s in data["segments"]] == ["hello", "world"]

    kwargs = fake_client.chat.completions.calls[0]
    assert kwargs["model"] == "fake-model"
    assert kwargs["response_format"] == {"type": "json_object"}


def test_translate_is_idempotent_unless_forced(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_transcript(job_dir)
    output_path = job_dir / "translation.ru.fake-model.json"
    output_path.write_text('{"already": "there"}')

    fake_client = FakeClient([])
    monkeypatch.setattr(translate, "build_openai_client", lambda cfg: fake_client)

    result_path = translate.run(
        "myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    assert result_path == output_path
    assert fake_client.chat.completions.calls == []


def test_translate_batches_with_overlap_context(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_transcript(job_dir, texts=["a", "b", "c", "d"])

    responses = [
        json.dumps({"translations": ["A", "B"]}),
        json.dumps({"translations": ["C", "D"]}),
    ]
    fake_client = FakeClient(responses)
    monkeypatch.setattr(translate, "build_openai_client", lambda cfg: fake_client)

    config = make_config(batch_size=2, overlap=1)
    output_path = translate.run(
        "myvid", config, "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    data = json.loads(output_path.read_text())
    assert [s["text"] for s in data["segments"]] == ["A", "B", "C", "D"]

    assert len(fake_client.chat.completions.calls) == 2
    second_call_user_msg = fake_client.chat.completions.calls[1]["messages"][1]["content"]
    assert "b" in second_call_user_msg  # source text of overlap segment
    assert "B" in second_call_user_msg  # its translation, for continuity


def test_translate_glossary_terms_in_system_prompt(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_transcript(job_dir)
    glossary_path = tmp_path / "glossary.txt"
    glossary_path.write_text("loss\ninference\n")

    fake_client = FakeClient([json.dumps({"translations": ["привет", "мир"]})])
    monkeypatch.setattr(translate, "build_openai_client", lambda cfg: fake_client)

    config = make_config(glossary=glossary_path)
    translate.run("myvid", config, "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None)

    system_msg = fake_client.chat.completions.calls[0]["messages"][0]["content"]
    assert "loss" in system_msg
    assert "inference" in system_msg


def test_translate_mismatched_translation_count_raises(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_transcript(job_dir)

    fake_client = FakeClient([json.dumps({"translations": ["only one"]})])
    monkeypatch.setattr(translate, "build_openai_client", lambda cfg: fake_client)

    with pytest.raises(translate.TranslateError):
        translate.run(
            "myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
        )


def test_translate_missing_transcript_raises(tmp_path, monkeypatch):
    (tmp_path / "jobs" / "myvid").mkdir(parents=True)
    fake_client = FakeClient([])
    monkeypatch.setattr(translate, "build_openai_client", lambda cfg: fake_client)

    with pytest.raises(GlamError):
        translate.run(
            "myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
        )


def test_translate_ambiguous_transcripts_raises(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_transcript(job_dir, asr_model="model-a")
    write_transcript(job_dir, asr_model="model-b")
    fake_client = FakeClient([])
    monkeypatch.setattr(translate, "build_openai_client", lambda cfg: fake_client)

    with pytest.raises(GlamError):
        translate.run(
            "myvid", make_config(), "ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
        )


def test_translate_explicit_transcript_path(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "myvid"
    write_transcript(job_dir, asr_model="model-a")
    explicit_path = write_transcript(job_dir, asr_model="model-b")

    fake_client = FakeClient([json.dumps({"translations": ["привет", "мир"]})])
    monkeypatch.setattr(translate, "build_openai_client", lambda cfg: fake_client)

    translate.run(
        "myvid", make_config(), "ru",
        jobs_root=tmp_path / "jobs",
        transcript_path=explicit_path,
        echo=lambda *_: None,
    )
    assert fake_client.chat.completions.calls  # ran without ambiguity error


def test_load_glossary_missing_file_raises(tmp_path):
    with pytest.raises(GlossaryError):
        load_glossary(tmp_path / "does-not-exist.txt")


def test_load_glossary_none_returns_empty():
    assert load_glossary(None) == []


def test_load_glossary_parses_plain_word_list(tmp_path):
    path = tmp_path / "glossary.txt"
    path.write_text(
        "# comment line\n"
        "loss\n"
        "\n"
        "inference  # trailing comment\n"
        "  gradient  \n"
    )
    assert load_glossary(path) == ["loss", "inference", "gradient"]
