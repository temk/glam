"""Pipeline configuration: the shared config file and its service definitions."""

import yaml
import dacite
from enum import Enum
from pathlib import Path
from dataclasses import field, dataclass

from glam.common.errors import GlamError

DEFAULT_CONFIG_PATH = "~/.glam.yaml"
DEFAULT_JOBS_PATH = "/tmp/glam_jobs"
DEFAULT_HOOK_TIMEOUT = 180  # seconds; 3 minutes


class ServiceName(str, Enum):
    """Remote model-backed services that may appear in the config's `services` list."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    TTS = "tts"


class Protocol(str, Enum):
    """How a service is called; selects the backend and the shape of `ServiceConfig.params`."""

    OPENAI = "openai"
    CHATTERBOX = "chatterbox"


class ConfigError(GlamError):
    pass


@dataclass
class HookConfig:
    url: str
    method: str = "POST"
    timeout: int = DEFAULT_HOOK_TIMEOUT


@dataclass
class ServiceHooks:
    """Optional side-effect calls around a step (e.g. load/unload a GPU model). At least one of
    `pre`/`post` must be set, else the section is pointless (validated in `read_config`)."""

    pre: HookConfig | None = None
    post: HookConfig | None = None


@dataclass
class ServiceConfig:
    name: ServiceName
    protocol: Protocol
    url: str
    # Protocol-specific fields, kept opaque here; each backend deserializes them into its own config.
    params: dict = field(default_factory=dict)
    hooks: ServiceHooks | None = None


@dataclass
class Defaults:
    """Optional job-level defaults; `init` falls back to these when a flag is omitted."""

    source: str | None = None
    target: str | None = None


@dataclass
class Config:
    """Mirrors the schema in docs/architecture.md: `job_dir`, `services`, and optional `defaults`."""

    services: list[ServiceConfig] = field(default_factory=list)
    job_dir: Path = field(default_factory=lambda: Path(DEFAULT_JOBS_PATH))
    defaults: Defaults = field(default_factory=Defaults)

    def __getitem__(self, name: str) -> ServiceConfig:
        """Return the service with the given name, or raise if the config does not define it."""
        service = next((s for s in self.services if s.name == name), None)
        if service is None:
            raise ConfigError(f"service '{name}' is not defined in the config's 'services' list")
        return service


# `cast` turns YAML strings into the `ServiceName`/`Protocol` enums; `type_hooks` expands `~` in
# `job_dir`; `strict` rejects unknown keys so typos surface at load time. `params` stays a plain
# dict — dacite does not recurse into it, leaving each backend to validate its own fields.
_DACITE = dacite.Config(cast=[Enum], type_hooks={Path: lambda v: Path(v).expanduser()}, strict=True)


def read_config(path: str | Path) -> Config:
    try:
        text = Path(path).expanduser().read_text()
    except OSError:
        raise ConfigError(f"config file not found: {path}") from None
    try:
        data = yaml.safe_load(text) or {}
        config = dacite.from_dict(data_class=Config, data=data, config=_DACITE)
    except (yaml.YAMLError, dacite.DaciteError, ValueError, TypeError) as e:
        raise ConfigError(f"invalid config file {path}: {e}") from e
    _validate_hooks(config)
    return config


def _validate_hooks(config: Config) -> None:
    for service in config.services:
        if service.hooks is not None and service.hooks.pre is None and service.hooks.post is None:
            raise ConfigError(f"service '{service.name.value}': 'hooks' must define at least one of 'pre'/'post'")
