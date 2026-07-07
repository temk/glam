import json
import pytest
from pathlib import Path
from click.testing import CliRunner

import glam.media as media_mod
from glam.cli import main
from glam.steps import init as init_step
from glam.common.job import read_job_manifest
from glam.steps.init import GlossaryError, slugify, _load_glossary_map


def _make_video(tmp_path: Path, name: str = "input.mp4") -> Path:
    video = tmp_path / name
    video.write_bytes(b"not-a-real-video")
    return video


@pytest.fixture
def fake_media(monkeypatch):
    """Stub the ffmpeg-backed media calls so init tests need no real video/ffmpeg."""
    calls = {"extract": 0, "probe": 0}

    def fake_extract_audio(src, dst, sample_rate=16000):
        calls["extract"] += 1
        Path(dst).write_bytes(b"fake-wav")

    def fake_probe_duration(src):
        calls["probe"] += 1
        return 12.5

    monkeypatch.setattr(media_mod, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(media_mod, "probe_duration", fake_probe_duration)
    return calls


# --- slugify ---


def test_slugify_basic():
    assert slugify("My Talk") == "my-talk"


def test_slugify_strips_and_collapses_non_alnum():
    assert slugify("  Hello, World!!  ") == "hello-world"


def test_slugify_non_ascii_falls_back():
    assert slugify("Привет") == "job"


def test_slugify_empty_falls_back():
    assert slugify("") == "job"


# --- _load_glossary_map ---


def test_glossary_json_array_becomes_identity_map(tmp_path):
    path = tmp_path / "g.json"
    path.write_text('["Kubernetes", "pod"]')
    assert _load_glossary_map(path) == {"Kubernetes": "Kubernetes", "pod": "pod"}


def test_glossary_json_object_kept_as_is(tmp_path):
    path = tmp_path / "g.json"
    path.write_text('{"pod": "под", "node": "нода"}')
    assert _load_glossary_map(path) == {"pod": "под", "node": "нода"}


def test_glossary_text_one_term_per_line_ignoring_comments_and_blanks(tmp_path):
    path = tmp_path / "g.txt"
    path.write_text("# header\nKubernetes\n\n  pod  \n# skip me\nCRD\n")
    assert _load_glossary_map(path) == {"Kubernetes": "Kubernetes", "pod": "pod", "CRD": "CRD"}


def test_glossary_detection_is_by_extension_not_content(tmp_path):
    path = tmp_path / "g.txt"  # JSON-looking content, but .txt -> read as one text line
    path.write_text('["A", "B"]')
    assert _load_glossary_map(path) == {'["A", "B"]': '["A", "B"]'}


@pytest.mark.parametrize(
    "content",
    [
        "{oops",  # malformed JSON
        '"scalar"',  # top-level neither object nor array
        "[1, 2]",  # array with non-string entries
        '{"a": 1}',  # object with non-string value
    ],
)
def test_glossary_invalid_json_raises(tmp_path, content):
    path = tmp_path / "g.json"
    path.write_text(content)
    with pytest.raises(GlossaryError):
        _load_glossary_map(path)


# --- run(): orchestration (media stubbed) ---


def test_run_creates_all_artifacts(tmp_path, fake_media):
    video = _make_video(tmp_path)
    job_path = init_step.run(
        video, source_lang="en", target_lang="ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    assert job_path == tmp_path / "jobs" / "input"
    assert (job_path / "source.mp4").is_symlink()
    assert (job_path / "source.mp4").resolve() == video.resolve()
    assert (job_path / "audio.wav").exists()
    assert (job_path / "glossary.json").exists()
    assert (job_path / "job.yaml").exists()


def test_run_writes_valid_manifest(tmp_path, fake_media):
    video = _make_video(tmp_path)
    job_path = init_step.run(
        video, source_lang="en", target_lang="ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )

    manifest = read_job_manifest(job_path / "job.yaml")
    assert manifest.version == 1
    assert manifest.job.id == "input"
    assert manifest.job.created_at
    assert manifest.source.original_path == str(video.resolve())
    assert manifest.source.filename == "input.mp4"
    assert manifest.source.artifact == "source.mp4"
    assert manifest.source.audio_artifact == "audio.wav"
    assert manifest.source.duration_seconds == 12.5
    assert manifest.languages.source == "en"
    assert manifest.languages.target == "ru"


def test_run_stores_voice_in_manifest(tmp_path, fake_media):
    video = _make_video(tmp_path)
    job_path = init_step.run(
        video, source_lang="en", target_lang="ru", voice="nova", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )
    assert read_job_manifest(job_path / "job.yaml").voice == "nova"


def test_run_voice_defaults_to_none(tmp_path, fake_media):
    video = _make_video(tmp_path)
    job_path = init_step.run(
        video, source_lang="en", target_lang="ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )
    assert read_job_manifest(job_path / "job.yaml").voice is None


def test_run_generates_job_id_from_filename(tmp_path, fake_media):
    video = _make_video(tmp_path, "My Talk.mp4")
    job_path = init_step.run(
        video, source_lang="en", target_lang="ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )
    assert job_path.name == "my-talk"


def test_run_prints_generated_job_id(tmp_path, fake_media):
    video = _make_video(tmp_path, "My Talk.mp4")
    logs: list[str] = []
    init_step.run(video, source_lang="en", target_lang="ru", jobs_root=tmp_path / "jobs", echo=logs.append)
    assert any("my-talk" in line for line in logs)


def test_run_uses_explicit_job_id(tmp_path, fake_media):
    video = _make_video(tmp_path)
    job_path = init_step.run(
        video,
        source_lang="en",
        target_lang="ru",
        job_id="custom-id",
        jobs_root=tmp_path / "jobs",
        echo=lambda *_: None,
    )
    assert job_path.name == "custom-id"


def test_run_writes_empty_glossary_when_none_given(tmp_path, fake_media):
    video = _make_video(tmp_path)
    job_path = init_step.run(
        video, source_lang="en", target_lang="ru", jobs_root=tmp_path / "jobs", echo=lambda *_: None
    )
    assert json.loads((job_path / "glossary.json").read_text()) == {}


def test_run_normalizes_provided_glossary(tmp_path, fake_media):
    video = _make_video(tmp_path)
    glossary = tmp_path / "g.json"
    glossary.write_text('["pod"]')
    job_path = init_step.run(
        video,
        source_lang="en",
        target_lang="ru",
        glossary_path=glossary,
        jobs_root=tmp_path / "jobs",
        echo=lambda *_: None,
    )
    assert json.loads((job_path / "glossary.json").read_text()) == {"pod": "pod"}


def test_run_is_idempotent(tmp_path, fake_media):
    video = _make_video(tmp_path)
    jobs = tmp_path / "jobs"
    init_step.run(video, source_lang="en", target_lang="ru", jobs_root=jobs, echo=lambda *_: None)
    init_step.run(video, source_lang="en", target_lang="ru", jobs_root=jobs, echo=lambda *_: None)
    assert fake_media["extract"] == 1  # second run skipped, not rebuilt


def test_run_force_rebuilds(tmp_path, fake_media):
    video = _make_video(tmp_path)
    jobs = tmp_path / "jobs"
    init_step.run(video, source_lang="en", target_lang="ru", jobs_root=jobs, echo=lambda *_: None)
    init_step.run(video, source_lang="en", target_lang="ru", jobs_root=jobs, force=True, echo=lambda *_: None)
    assert fake_media["extract"] == 2
    assert (jobs / "input" / "source.mp4").is_symlink()


# --- CLI ---


def test_cli_errors_when_language_missing_and_no_defaults(tmp_path, fake_media):
    video = _make_video(tmp_path)
    config = tmp_path / "glam.yaml"
    config.write_text(f"job_dir: {tmp_path / 'jobs'}\n")  # no defaults section

    result = CliRunner().invoke(main, ["init", str(video), "--target", "ru", "--config", str(config)])
    assert result.exit_code != 0
    assert "--source" in result.output


def test_cli_uses_defaults_for_languages(tmp_path, fake_media):
    video = _make_video(tmp_path)
    jobs = tmp_path / "jobs"
    config = tmp_path / "glam.yaml"
    config.write_text(f"job_dir: {jobs}\ndefaults:\n  source: en\n  target: ru\n")

    result = CliRunner().invoke(main, ["init", str(video), "--config", str(config)])
    assert result.exit_code == 0, result.output
    manifest = read_job_manifest(jobs / "input" / "job.yaml")
    assert manifest.languages.source == "en"
    assert manifest.languages.target == "ru"


def test_cli_flag_overrides_default_language(tmp_path, fake_media):
    video = _make_video(tmp_path)
    jobs = tmp_path / "jobs"
    config = tmp_path / "glam.yaml"
    config.write_text(f"job_dir: {jobs}\ndefaults:\n  source: en\n  target: ru\n")

    result = CliRunner().invoke(main, ["init", str(video), "--target", "de", "--config", str(config)])
    assert result.exit_code == 0, result.output
    manifest = read_job_manifest(jobs / "input" / "job.yaml")
    assert manifest.languages.source == "en"  # from defaults
    assert manifest.languages.target == "de"  # flag overrides default


def test_cli_reads_job_dir_from_config(tmp_path, fake_media):
    video = _make_video(tmp_path)
    jobs = tmp_path / "configured-jobs"
    config = tmp_path / "glam.yaml"
    config.write_text(f"job_dir: {jobs}\n")

    result = CliRunner().invoke(main, ["init", str(video), "--source", "en", "--target", "ru", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert (jobs / "input" / "job.yaml").exists()


def test_cli_passes_job_id_and_glossary(tmp_path, fake_media):
    video = _make_video(tmp_path)
    jobs = tmp_path / "configured-jobs"
    config = tmp_path / "glam.yaml"
    config.write_text(f"job_dir: {jobs}\n")
    glossary = tmp_path / "g.txt"
    glossary.write_text("pod\nnode\n")

    result = CliRunner().invoke(
        main,
        [
            "init",
            str(video),
            "--source",
            "en",
            "--target",
            "ru",
            "--job-id",
            "jid",
            "--glossary",
            str(glossary),
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads((jobs / "jid" / "glossary.json").read_text()) == {"pod": "pod", "node": "node"}
