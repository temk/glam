from openai import OpenAI

from glam.config import resolve_api_key


def build_openai_client(step_cfg):
    return OpenAI(base_url=step_cfg["base_url"], api_key=resolve_api_key(step_cfg))
