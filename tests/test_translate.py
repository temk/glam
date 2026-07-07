import json
import openai
import pytest
from types import SimpleNamespace
from pathlib import Path

from glam.steps import translate as translate_step
from glam.common.job import JobInfo, Languages, SourceInfo, JobManifest, write_job_manifest
from glam.common.config import Config, Protocol, ConfigError, ServiceName, ServiceConfig
from glam.steps.translate import TranslateError
from glam.backend.translate.base import TranslateBackendError

TRANSCRIPT = {
    "version": 1,
    "step": "transcribe",
    "job_id": "jobA",
    "source_language": "en",
    "model": "whisper-x",
    "audio_artifact": "audio.wav",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello."},
        {"id": 1, "start": 1.5, "end": 3.0, "text": "World."},
    ],
}


def _make_job(
    tmp_path: Path,
    job_id: str = "jobA",
    transcript: dict | None = TRANSCRIPT,
    glossary: object = None,
) -> Path:
    job_path = tmp_path / job_id
    job_path.mkdir(parents=True)
    manifest = JobManifest(
        version=1,
        job=JobInfo(id=job_id, created_at="2026-07-05T00:00:00+00:00"),
        source=SourceInfo(
            original_path="/x/video.mp4",
            filename="video.mp4",
            artifact="source.mp4",
            audio_artifact="audio.wav",
            duration_seconds=3.0,
        ),
        languages=Languages(source="en", target="ru"),
    )
    write_job_manifest(manifest, job_path / "job.yaml")
    if transcript is not None:
        (job_path / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2) + "\n")
    if glossary is not None:
        (job_path / "glossary.json").write_text(json.dumps(glossary, ensure_ascii=False))
    else:
        (job_path / "glossary.json").write_text("{}")
    return job_path


def _config(tmp_path: Path, with_service: bool = True, model: str = "llm-x") -> Config:
    services = []
    if with_service:
        services.append(
            ServiceConfig(
                name=ServiceName.TRANSLATE, protocol=Protocol.OPENAI, url="http://llm/v1", params={"model": model}
            )
        )
    return Config(services=services, job_dir=tmp_path)


def _reply(segments: list[dict]) -> str:
    return json.dumps({"segments": segments})


def _auto_reply(kwargs) -> str:
    """Echo a `tr:<text>` translation for each requested 'translate' segment."""
    payload = json.loads(kwargs["messages"][1]["content"])
    return _reply([{"id": s["id"], "translated_text": f"tr:{s['text']}"} for s in payload["translate"]])


def _fake_client(calls, reply=None, error=None):
    def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        content = reply if reply is not None else _auto_reply(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture
def patch_client(monkeypatch):
    """Replace the LLM client factory; return a recorder for the requests it receives."""

    def install(reply=None, error=None):
        calls: list[dict] = []
        client = _fake_client(calls, reply=reply, error=error)
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
        return calls

    return install


def test_creates_translation(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path)
    path = translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    data = json.loads(path.read_text())
    assert path.name == "translation.ru.json"
    assert [s["translated_text"] for s in data["segments"]] == ["tr:Hello.", "tr:World."]


def test_target_override_changes_artifact_name(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path)  # job.yaml target is ru
    path = translate_step.run("jobA", _config(tmp_path), target="de", echo=lambda *_: None)

    assert path.name == "translation.de.json"


def test_translation_equals_transcript_without_translated_text(tmp_path, patch_client):
    patch_client()
    job_path = _make_job(tmp_path)
    path = translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    data = json.loads(path.read_text())
    for segment in data["segments"]:
        assert "translated_text" in segment
        del segment["translated_text"]
    assert data == json.loads((job_path / "transcript.json").read_text())


def test_preserves_segment_ids_order_timestamps_and_count(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path)
    path = translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    data = json.loads(path.read_text())
    assert len(data["segments"]) == len(TRANSCRIPT["segments"])
    for out, src in zip(data["segments"], TRANSCRIPT["segments"]):
        assert out["id"] == src["id"]
        assert out["start"] == src["start"]
        assert out["end"] == src["end"]


def test_uses_translate_model_and_json_response_format(tmp_path, patch_client):
    calls = patch_client()
    _make_job(tmp_path)
    translate_step.run("jobA", _config(tmp_path, model="llm-x"), echo=lambda *_: None)

    assert len(calls) == 1
    assert calls[0]["model"] == "llm-x"
    assert calls[0]["response_format"]["type"] == "json_schema"


def test_glossary_terms_appear_in_prompt(tmp_path, patch_client):
    calls = patch_client()
    _make_job(tmp_path, glossary={"tensor": "тензор", "GPU": "GPU"})
    translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    system_prompt = calls[0]["messages"][0]["content"]
    assert "tensor" in system_prompt
    assert "тензор" in system_prompt
    assert "GPU" in system_prompt


def test_empty_glossary_translates_normally(tmp_path, patch_client):
    calls = patch_client()
    _make_job(tmp_path, glossary={})
    translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert len(calls) == 1  # translation still runs


def test_batches_with_translated_context(tmp_path, patch_client):
    segments = [{"id": i, "start": float(i), "end": float(i) + 1, "text": f"s{i}"} for i in range(25)]
    transcript = {**TRANSCRIPT, "segments": segments}
    calls = patch_client()
    _make_job(tmp_path, transcript=transcript)

    path = translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None, batch_size=10, context_size=3)
    data = json.loads(path.read_text())

    assert len(calls) == 3  # 10 + 10 + 5
    first = json.loads(calls[0]["messages"][1]["content"])
    assert first["context"] == ""  # nothing precedes the first batch
    assert [s["id"] for s in first["translate"]] == list(range(10))

    second = json.loads(calls[1]["messages"][1]["content"])
    assert second["context"] == "tr:s7 tr:s8 tr:s9"  # 3 preceding translations as plain text
    assert [s["id"] for s in second["translate"]] == list(range(10, 20))

    assert len(data["segments"]) == 25
    assert all(s["translated_text"] for s in data["segments"])


def test_retries_missing_segments(tmp_path, monkeypatch):
    _make_job(tmp_path)  # two segments, ids 0 and 1
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        ids = [s["id"] for s in json.loads(kwargs["messages"][1]["content"])["translate"]]
        if len(calls) == 1:
            ids = [i for i in ids if i != 1]  # first round drops one segment
        content = _reply([{"id": i, "translated_text": f"tr{i}"} for i in ids])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)

    path = translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
    data = json.loads(path.read_text())

    assert len(calls) == 2  # the dropped segment triggered a second request
    assert [s["id"] for s in json.loads(calls[1]["messages"][1]["content"])["translate"]] == [1]  # only the straggler
    assert [s["translated_text"] for s in data["segments"]] == ["tr0", "tr1"]


def test_dump_writes_one_file_per_batch(tmp_path, patch_client):
    segments = [{"id": i, "start": float(i), "end": float(i) + 1, "text": f"s{i}"} for i in range(25)]
    patch_client()
    job_path = _make_job(tmp_path, transcript={**TRANSCRIPT, "segments": segments})

    translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None, dump=True, batch_size=10)

    dump_dir = job_path / "translate.ru.dump"
    assert not (dump_dir / "004.json").exists()  # only 3 batches for 25 segments
    batch2 = json.loads((dump_dir / "002.json").read_text())
    assert isinstance(batch2, list) and len(batch2) == 1
    entry = batch2[0]
    assert entry["requested_ids"] == list(range(10, 20))  # this file holds only its batch
    assert entry["request"]["messages"][0]["role"] == "system"
    assert "content" in entry["response"]


def test_dump_records_every_retry_round(tmp_path, monkeypatch):
    _make_job(tmp_path)  # two segments, ids 0 and 1
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        ids = [s["id"] for s in json.loads(kwargs["messages"][1]["content"])["translate"]]
        if len(calls) == 1:
            ids = [i for i in ids if i != 1]  # first round drops one → forces a retry
        content = _reply([{"id": i, "translated_text": f"tr{i}"} for i in ids])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)

    translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None, dump=True)

    records = json.loads((tmp_path / "jobA" / "translate.ru.dump" / "001.json").read_text())
    assert len(records) == 2  # both the initial request and the retry are dumped


def test_no_dump_file_by_default(tmp_path, patch_client):
    patch_client()
    job_path = _make_job(tmp_path)
    translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert not (job_path / "translate.ru.dump").exists()


def test_reports_progress(tmp_path, patch_client):
    segments = [{"id": i, "start": float(i), "end": float(i) + 1, "text": f"s{i}"} for i in range(25)]
    patch_client()
    _make_job(tmp_path, transcript={**TRANSCRIPT, "segments": segments})

    messages: list[str] = []
    translate_step.run("jobA", _config(tmp_path), echo=messages.append, batch_size=10)

    assert any("translating segments 1-10 of 25" in m for m in messages)
    assert any("translating segments 21-25 of 25" in m for m in messages)


def test_skips_when_translation_exists(tmp_path, patch_client):
    calls = patch_client()
    job_path = _make_job(tmp_path)
    (job_path / "translation.ru.json").write_text('{"existing": true}')

    translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)

    assert calls == []  # LLM was not called
    assert json.loads((job_path / "translation.ru.json").read_text()) == {"existing": True}


def test_force_recreates(tmp_path, patch_client):
    calls = patch_client()
    job_path = _make_job(tmp_path)
    (job_path / "translation.ru.json").write_text('{"existing": true}')

    translate_step.run("jobA", _config(tmp_path), force=True, echo=lambda *_: None)

    assert len(calls) == 1
    assert "segments" in json.loads((job_path / "translation.ru.json").read_text())


def test_job_not_found(tmp_path, patch_client):
    patch_client()
    with pytest.raises(TranslateError):
        translate_step.run("missing", _config(tmp_path), echo=lambda *_: None)


def test_missing_transcript(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path, transcript=None)
    with pytest.raises(TranslateError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_missing_glossary(tmp_path, patch_client):
    patch_client()
    job_path = _make_job(tmp_path)
    (job_path / "glossary.json").unlink()
    with pytest.raises(TranslateError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_invalid_glossary_format(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path, glossary=["not", "a", "map"])
    with pytest.raises(TranslateError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_invalid_transcript_format(tmp_path, patch_client):
    patch_client()
    job_path = _make_job(tmp_path)
    (job_path / "transcript.json").write_text('{"version": 1}')  # missing required fields
    with pytest.raises(TranslateError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_missing_service_in_config(tmp_path, patch_client):
    patch_client()
    _make_job(tmp_path)
    with pytest.raises(ConfigError):
        translate_step.run("jobA", _config(tmp_path, with_service=False), echo=lambda *_: None)


def test_service_unavailable_raises(tmp_path, patch_client):
    patch_client(error=openai.OpenAIError("connection refused"))
    _make_job(tmp_path)
    with pytest.raises(TranslateBackendError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_invalid_backend_response_raises(tmp_path, patch_client):
    patch_client(reply="not json")
    _make_job(tmp_path)
    with pytest.raises(TranslateError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_missing_segment_id_in_response_raises(tmp_path, patch_client):
    patch_client(reply=_reply([{"id": 0, "translated_text": "Привет."}]))  # id 1 missing
    _make_job(tmp_path)
    with pytest.raises(TranslateError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_duplicate_segment_id_in_response_raises(tmp_path, patch_client):
    patch_client(reply=_reply([{"id": 0, "translated_text": "a"}, {"id": 0, "translated_text": "b"}]))
    _make_job(tmp_path)
    with pytest.raises(TranslateError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)


def test_unknown_segment_id_in_response_raises(tmp_path, patch_client):
    patch_client(reply=_reply([{"id": 0, "translated_text": "a"}, {"id": 99, "translated_text": "b"}]))
    _make_job(tmp_path)
    with pytest.raises(TranslateError):
        translate_step.run("jobA", _config(tmp_path), echo=lambda *_: None)
