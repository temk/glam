import pytest
from pathlib import Path

from glam.common.config import Protocol, ConfigError, ServiceName, read_config

VALID_CONFIG = """
job_dir: /tmp/jobs

services:
  - name: transcribe
    protocol: openai
    url: http://h:8000/v1
    params:
      model: whisper
  - name: translate
    protocol: openai
    url: http://h:11434/v1
    params:
      model: qwen
      api_key: secret
  - name: tts
    protocol: chatterbox
    url: http://h:8004
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "glam.yaml"
    path.write_text(text)
    return path


def test_config_parses_valid_file(tmp_path):
    config = read_config(_write(tmp_path, VALID_CONFIG))

    assert config.job_dir == Path("/tmp/jobs")
    assert [s.name for s in config.services] == [ServiceName.TRANSCRIBE, ServiceName.TRANSLATE, ServiceName.TTS]

    translate = config["translate"]
    assert translate.protocol is Protocol.OPENAI
    assert translate.url == "http://h:11434/v1"
    assert translate.params == {"model": "qwen", "api_key": "secret"}

    tts = config["tts"]
    assert tts.protocol is Protocol.CHATTERBOX
    assert tts.params == {}  # protocol-specific bag defaults to empty


def test_job_dir_expands_user(tmp_path):
    config = read_config(_write(tmp_path, "job_dir: ~/jobs\nservices: []\n"))
    assert config.job_dir == Path.home() / "jobs"


def test_missing_service_raises(tmp_path):
    config = read_config(_write(tmp_path, VALID_CONFIG))
    with pytest.raises(ConfigError):
        config["nope"]


def test_defaults_section_parsed(tmp_path):
    config = read_config(_write(tmp_path, "defaults:\n  source: en\n  target: ru\nservices: []\n"))
    assert config.defaults.source == "en"
    assert config.defaults.target == "ru"


def test_defaults_absent_gives_empty(tmp_path):
    config = read_config(_write(tmp_path, "services: []\n"))
    assert config.defaults.source is None
    assert config.defaults.target is None


@pytest.mark.parametrize(
    "text",
    [
        # missing required protocol
        "services:\n  - name: tts\n    url: http://h/v1\n",
        # missing required url
        "services:\n  - name: tts\n    protocol: openai\n",
        # unknown service name
        "services:\n  - name: asr\n    protocol: openai\n    url: u\n",
        # unknown protocol
        "services:\n  - name: tts\n    protocol: bogus\n    url: u\n",
        # protocol-specific field left at the top level instead of under params (strict rejects it)
        "services:\n  - name: tts\n    protocol: chatterbox\n    url: u\n    model: x\n",
        # params is not a mapping
        "services:\n  - name: tts\n    protocol: chatterbox\n    url: u\n    params: nope\n",
        # unknown key in the defaults section (strict)
        "defaults:\n  lang: en\n",
    ],
)
def test_config_rejects_invalid_file(tmp_path, text):
    with pytest.raises(ConfigError):
        read_config(_write(tmp_path, text))
