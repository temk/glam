import pytest
from pathlib import Path

from glam.common.config import DEFAULT_HOOK_TIMEOUT, Protocol, ConfigError, ServiceName, read_config

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


def test_hooks_parsed_with_defaults(tmp_path):
    text = (
        "services:\n"
        "  - name: translate\n"
        "    protocol: openai\n"
        "    url: http://h/v1\n"
        "    hooks:\n"
        "      pre:\n"
        "        url: http://h:9090/prepare\n"
        "        timeout: 120\n"
        "      post:\n"
        "        url: http://h:9090/release\n"
    )
    config = read_config(_write(tmp_path, text))
    hooks = config["translate"].hooks

    assert hooks is not None
    assert hooks.pre.url == "http://h:9090/prepare"
    assert hooks.pre.method == "POST"  # default method
    assert hooks.pre.timeout == 120
    assert hooks.post.method == "POST"
    assert hooks.post.timeout == DEFAULT_HOOK_TIMEOUT  # default timeout


def test_hooks_absent_is_none(tmp_path):
    assert read_config(_write(tmp_path, VALID_CONFIG))["translate"].hooks is None


def test_empty_hooks_section_rejected(tmp_path):
    text = "services:\n  - name: tts\n    protocol: chatterbox\n    url: u\n    hooks: {}\n"
    with pytest.raises(ConfigError, match="at least one of 'pre'/'post'"):
        read_config(_write(tmp_path, text))


def test_hook_without_url_rejected(tmp_path):
    text = (
        "services:\n  - name: tts\n    protocol: chatterbox\n    url: u\n    hooks:\n      pre:\n        timeout: 30\n"
    )
    with pytest.raises(ConfigError):
        read_config(_write(tmp_path, text))


def test_job_dir_expands_user(tmp_path):
    config = read_config(_write(tmp_path, "job_dir: ~/jobs\nservices: []\n"))
    assert config.job_dir == Path.home() / "jobs"


def test_missing_service_raises(tmp_path):
    config = read_config(_write(tmp_path, VALID_CONFIG))
    with pytest.raises(ConfigError):
        config["nope"]


def test_defaults_section_parsed(tmp_path):
    config = read_config(_write(tmp_path, "defaults:\n  source: en\n  target: ru\n  glossary: ./g.txt\nservices: []\n"))
    assert config.defaults.source == "en"
    assert config.defaults.target == "ru"
    assert config.defaults.glossary == "./g.txt"


def test_defaults_absent_gives_empty(tmp_path):
    config = read_config(_write(tmp_path, "services: []\n"))
    assert config.defaults.source is None
    assert config.defaults.target is None
    assert config.defaults.glossary is None


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
