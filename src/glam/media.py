import ffmpeg

from glam.errors import GlamError


class MediaError(GlamError):
    pass


def ffprobe_json(path):
    try:
        return ffmpeg.probe(str(path))
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise MediaError(f"ffprobe failed on {path}: {stderr.strip()}") from e


def probe_duration(path):
    return float(ffprobe_json(path)["format"]["duration"])


def extract_audio(src, dst, sample_rate=16000):
    try:
        (
            ffmpeg
            .input(str(src))
            .audio
            .output(str(dst), ac=1, ar=sample_rate)
            .run(overwrite_output=True, quiet=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise MediaError(f"ffmpeg audio extraction failed on {src}: {stderr.strip()}") from e
