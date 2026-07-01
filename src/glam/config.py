import os
from pathlib import Path

import yaml

from glam.errors import GlamError


class ConfigError(GlamError):
    pass


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def step_config(config, step_name):
    try:
        return config["steps"][step_name]
    except KeyError:
        raise ConfigError(f"missing 'steps.{step_name}' in config") from None


def resolve_api_key(step_cfg):
    env_var = step_cfg.get("api_key_env")
    if not env_var:
        return "not-needed"
    key = os.environ.get(env_var)
    if not key:
        raise ConfigError(f"environment variable '{env_var}' is not set")
    return key
