import json
from pathlib import Path

import openai

from glam.clients import build_openai_client
from glam.config import step_config
from glam.errors import GlamError
from glam.paths import job_dir, slugify


class TranscribeError(GlamError):
    pass


def run(video_id, config, jobs_root=Path("jobs"), force=False, echo=print):
    job_path = job_dir(jobs_root, video_id)
    audio_path = job_path / "audio.wav"
    if not audio_path.exists():
        raise TranscribeError(f"{audio_path} not found — run 'glam init' for this job first")

    asr_cfg = step_config(config, "asr")
    model = asr_cfg["model"]
    granularities = asr_cfg.get("timestamp_granularities", ["segment", "word"])

    transcript_path = job_path / f"transcript.{slugify(model)}.json"
    if transcript_path.exists() and not force:
        echo(f"skip transcription, already exists: {transcript_path}")
        return transcript_path

    client = build_openai_client(asr_cfg)
    try:
        with audio_path.open("rb") as f:
            response = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=granularities,
            )
    except openai.OpenAIError as e:
        raise TranscribeError(
            f"ASR request to {asr_cfg.get('base_url')} failed: {e}. "
            "Verify the backend is reachable and speaks the OpenAI-compatible "
            "/v1/audio/transcriptions shape before assuming this is a code bug."
        ) from e

    data = response.model_dump()
    data["model"] = model
    transcript_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    echo(f"wrote {transcript_path}")
    return transcript_path
