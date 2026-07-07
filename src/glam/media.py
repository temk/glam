import json
import subprocess

from glam.common.errors import GlamError


class MediaError(GlamError):
    pass


def run_ffmpeg(args, error_prefix, error_cls=MediaError):
    """Run `ffmpeg -y <args>`, raising error_cls with stderr on failure."""
    cmd = ["ffmpeg", "-y", *[str(a) for a in args]]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except OSError as e:
        raise error_cls(f"{error_prefix}: {e}") from e
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip() if result.stderr else ""
        raise error_cls(f"{error_prefix}: {stderr}")
    return result


def ffprobe_json(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except OSError as e:
        raise MediaError(f"ffprobe failed on {path}: {e}") from e
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip() if result.stderr else ""
        raise MediaError(f"ffprobe failed on {path}: {stderr}")
    return json.loads(result.stdout)


def probe_duration(path):
    return float(ffprobe_json(path)["format"]["duration"])


def extract_audio(src, dst, sample_rate=16000):
    run_ffmpeg(
        ["-i", src, "-vn", "-ac", "1", "-ar", str(sample_rate), dst],
        f"ffmpeg audio extraction failed on {src}",
    )
