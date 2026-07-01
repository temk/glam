import json
import tempfile
import wave
from pathlib import Path

import ffmpeg
import openai

from glam import media
from glam.clients import build_openai_client
from glam.config import step_config
from glam.errors import GlamError
from glam.paths import job_dir, resolve_artifact, slugify

MIN_SEGMENT_DURATION = 0.3
SILENCE_DURATION = 0.2


class TTSError(GlamError):
    pass


def run(video_id, config, lang, jobs_root=Path("jobs"), translation_path=None, force=False, echo=print):
    job_path = job_dir(jobs_root, video_id)
    tts_cfg = step_config(config, "tts")
    model = tts_cfg["model"]
    voice = tts_cfg.get("voice", "default")
    sample_rate = tts_cfg.get("sample_rate", 24000)
    max_atempo = tts_cfg.get("max_atempo", 1.3)
    response_format = tts_cfg.get("response_format", "wav")
    speed = tts_cfg.get("speed")

    model_slug = slugify(model)
    voice_slug = slugify(voice)
    track_path = job_path / f"tts_track.{lang}.{model_slug}.{voice_slug}.wav"
    segments_dir = job_path / f"tts_segments.{lang}.{model_slug}.{voice_slug}"

    if track_path.exists() and not force:
        echo(f"skip tts, already exists: {track_path}")
        return track_path

    translate_cfg = config.get("steps", {}).get("translate")
    exact_candidate = None
    if translate_cfg and translate_cfg.get("model"):
        exact_candidate = job_path / f"translation.{lang}.{slugify(translate_cfg['model'])}.json"
    translation_path = resolve_artifact(
        job_path, f"translation.{lang}.*.json", translation_path,
        exact_candidate=exact_candidate, hint=f"run 'glam translate --lang {lang}' first",
    )
    translation = json.loads(Path(translation_path).read_text())
    segments = translation.get("segments") or []
    if not segments:
        raise TTSError(f"no segments found in {translation_path}")

    segments_dir.mkdir(parents=True, exist_ok=True)
    client = build_openai_client(tts_cfg)

    clip_paths = []
    offsets = []
    for i, seg in enumerate(segments):
        clip_path = segments_dir / f"{i:04d}.wav"
        if force or not clip_path.exists():
            text = (seg.get("text") or "").strip()
            if text:
                target_duration = max(seg["end"] - seg["start"], MIN_SEGMENT_DURATION)
                _synthesize_segment(
                    client, model, voice, text, clip_path,
                    target_duration, sample_rate, max_atempo, response_format, speed,
                )
            else:
                _write_silence(clip_path, SILENCE_DURATION, sample_rate)
            echo(f"wrote {clip_path}")
        else:
            echo(f"skip segment clip, already exists: {clip_path}")
        clip_paths.append(clip_path)
        offsets.append(seg["start"])

    _assemble_track(clip_paths, offsets, track_path, sample_rate)
    echo(f"wrote {track_path}")
    return track_path


def _synthesize_segment(client, model, voice, text, clip_path, target_duration,
                         sample_rate, max_atempo, response_format, speed):
    kwargs = {"model": model, "voice": voice, "input": text, "response_format": response_format}
    if speed is not None:
        kwargs["speed"] = speed

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        raw_path = Path(tmp.name)
    try:
        try:
            response = client.audio.speech.create(**kwargs)
        except openai.OpenAIError as e:
            raise TTSError(
                f"TTS request failed: {e}. Verify the backend is reachable and speaks "
                "the OpenAI-compatible /v1/audio/speech shape before assuming this is a code bug."
            ) from e
        response.stream_to_file(str(raw_path))
        _finalize_clip(raw_path, clip_path, target_duration, sample_rate, max_atempo)
    finally:
        raw_path.unlink(missing_ok=True)


def _finalize_clip(raw_path, out_path, target_duration, sample_rate, max_atempo):
    actual_duration = media.probe_duration(raw_path)
    factor = actual_duration / target_duration if target_duration > 0 else 1.0
    factor = min(max(factor, 1 / max_atempo), max_atempo)

    stream = ffmpeg.input(str(raw_path)).audio
    if abs(factor - 1.0) > 0.01:
        stream = stream.filter("atempo", factor)
    try:
        (
            stream
            .output(str(out_path), ac=1, ar=sample_rate, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise TTSError(f"failed to sync clip duration for {raw_path}: {stderr.strip()}") from e


def _write_silence(path, duration, sample_rate):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"\x00" * int(duration * sample_rate) * 2)


def _assemble_track(clip_paths, offsets, output_path, sample_rate):
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        cursor = 0.0
        for clip_path, start in zip(clip_paths, offsets):
            gap = start - cursor
            if gap > 0:
                out.writeframes(b"\x00" * int(gap * sample_rate) * 2)
                cursor = start
            with wave.open(str(clip_path), "rb") as clip:
                out.writeframes(clip.readframes(clip.getnframes()))
                cursor += clip.getnframes() / clip.getframerate()
