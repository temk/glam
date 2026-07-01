import functools
from importlib.metadata import version as pkg_version
from pathlib import Path

import click

from glam import resegment as resegment_defaults
from glam.config import load_config
from glam.errors import GlamError
from glam.steps import init as init_step
from glam.steps import mux as mux_step
from glam.steps import subtitles as subtitles_step
from glam.steps import transcribe as transcribe_step
from glam.steps import translate as translate_step
from glam.steps import tts as tts_step


def handle_glam_errors(fn):
    """Report GlamError as a clean CLI message instead of a Python traceback."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except GlamError as e:
            raise click.ClickException(str(e)) from e
    return wrapper


@click.group()
def main():
    """GLAM — Glossary-Locked Audio Muxer."""


@main.command("version")
def version_cmd():
    """Print the glam version."""
    click.echo(pkg_version("glam"))


@main.command("init")
@click.argument("video_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--id", "video_id", default=None, help="Job id (default: derived from filename)")
@click.option("--force", is_flag=True, help="Recompute even if outputs already exist")
@click.option(
    "--jobs-dir",
    default="jobs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root directory for job data",
)
@handle_glam_errors
def init_cmd(video_file, video_id, force, jobs_dir):
    """Register a job from a local video file."""
    job = init_step.run(
        video_file,
        video_id=video_id,
        jobs_root=Path(jobs_dir),
        force=force,
        echo=click.echo,
    )
    click.echo(f"job '{job.video_id}' ready at {job.job_dir}")


@main.command("transcribe")
@click.argument("video_id")
@click.option(
    "-c", "--config", "config_path",
    default="conf/config.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config file",
)
@click.option("--force", is_flag=True, help="Recompute even if output already exists")
@click.option(
    "--jobs-dir",
    default="jobs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root directory for job data",
)
@handle_glam_errors
def transcribe_cmd(video_id, config_path, force, jobs_dir):
    """Transcribe a job's audio via the configured ASR backend."""
    config = load_config(config_path)
    transcript_path = transcribe_step.run(
        video_id,
        config,
        jobs_root=Path(jobs_dir),
        force=force,
        echo=click.echo,
    )
    click.echo(f"transcript ready at {transcript_path}")


@main.command("translate")
@click.argument("video_id")
@click.option("--lang", required=True, help="Target language code, e.g. ru")
@click.option(
    "-c", "--config", "config_path",
    default="conf/config.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config file",
)
@click.option(
    "--transcript", "transcript_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Explicit transcript JSON to translate (default: auto-detect from job dir)",
)
@click.option("--force", is_flag=True, help="Recompute even if output already exists")
@click.option(
    "--jobs-dir",
    default="jobs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root directory for job data",
)
@handle_glam_errors
def translate_cmd(video_id, lang, config_path, transcript_path, force, jobs_dir):
    """Translate a job's transcript via the configured LLM backend, applying the glossary."""
    config = load_config(config_path)
    output_path = translate_step.run(
        video_id,
        config,
        lang,
        jobs_root=Path(jobs_dir),
        transcript_path=transcript_path,
        force=force,
        echo=click.echo,
    )
    click.echo(f"translation ready at {output_path}")


@main.command("subtitles")
@click.argument("video_id")
@click.option("--lang", required=True, help="Target language code, e.g. ru")
@click.option(
    "--translation", "translation_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Explicit translation JSON (default: auto-detect from job dir)",
)
@click.option("--force", is_flag=True, help="Recompute even if output already exists")
@click.option(
    "--jobs-dir",
    default="jobs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root directory for job data",
)
@click.option("--cps", default=resegment_defaults.DEFAULT_CPS, show_default=True, help="Reading speed, chars/sec")
@click.option(
    "--max-chars-per-line",
    default=resegment_defaults.DEFAULT_MAX_CHARS_PER_LINE,
    show_default=True,
)
@click.option("--max-lines", default=resegment_defaults.DEFAULT_MAX_LINES, show_default=True)
@handle_glam_errors
def subtitles_cmd(video_id, lang, translation_path, force, jobs_dir, cps, max_chars_per_line, max_lines):
    """Generate .srt subtitles from a job's translation, re-timed for reading speed."""
    output_path = subtitles_step.run(
        video_id,
        lang,
        jobs_root=Path(jobs_dir),
        translation_path=translation_path,
        force=force,
        cps=cps,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        echo=click.echo,
    )
    click.echo(f"subtitles ready at {output_path}")


@main.command("tts")
@click.argument("video_id")
@click.option("--lang", required=True, help="Target language code, e.g. ru")
@click.option(
    "-c", "--config", "config_path",
    default="conf/config.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config file",
)
@click.option(
    "--translation", "translation_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Explicit translation JSON (default: auto-detect from job dir)",
)
@click.option("--force", is_flag=True, help="Recompute even if outputs already exist")
@click.option(
    "--jobs-dir",
    default="jobs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root directory for job data",
)
@handle_glam_errors
def tts_cmd(video_id, lang, config_path, translation_path, force, jobs_dir):
    """Synthesize a dubbed audio track for a job via the configured TTS backend."""
    config = load_config(config_path)
    track_path = tts_step.run(
        video_id,
        config,
        lang,
        jobs_root=Path(jobs_dir),
        translation_path=translation_path,
        force=force,
        echo=click.echo,
    )
    click.echo(f"tts track ready at {track_path}")


@main.command("mux")
@click.argument("video_id")
@click.option("--lang", required=True, help="Target language code, e.g. ru")
@click.option("--hardsub", is_flag=True, help="Burn subtitles into the video instead of a soft track")
@click.option(
    "--keep-original-audio/--no-keep-original-audio",
    default=True,
    help="Include the original-language audio track alongside the dubbed one",
)
@click.option(
    "--subtitles", "subtitles_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Explicit subtitles file (default: auto-detect from job dir)",
)
@click.option(
    "--tts-track", "tts_track_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Explicit dubbed audio track (default: auto-detect from job dir)",
)
@click.option("--force", is_flag=True, help="Recompute even if output already exists")
@click.option(
    "--jobs-dir",
    default="jobs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root directory for job data",
)
@handle_glam_errors
def mux_cmd(video_id, lang, hardsub, keep_original_audio, subtitles_path, tts_track_path, force, jobs_dir):
    """Mux source video, original + dubbed audio, and subtitles into the final output."""
    output_path = mux_step.run(
        video_id,
        lang,
        jobs_root=Path(jobs_dir),
        hardsub=hardsub,
        keep_original_audio=keep_original_audio,
        subtitles_path=subtitles_path,
        tts_track_path=tts_track_path,
        force=force,
        echo=click.echo,
    )
    click.echo(f"output ready at {output_path}")


@main.command("run")
@click.argument("video_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--lang", required=True, help="Target language code, e.g. ru")
@click.option("--id", "video_id", default=None, help="Job id (default: derived from filename)")
@click.option(
    "-c", "--config", "config_path",
    default="conf/config.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config file",
)
@click.option("--hardsub", is_flag=True, help="Burn subtitles into the video instead of a soft track")
@click.option(
    "--keep-original-audio/--no-keep-original-audio",
    default=True,
    help="Include the original-language audio track alongside the dubbed one",
)
@click.option("--force", is_flag=True, help="Recompute every step even if outputs already exist")
@click.option(
    "--jobs-dir",
    default="jobs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root directory for job data",
)
@handle_glam_errors
def run_cmd(video_file, lang, video_id, config_path, hardsub, keep_original_audio, force, jobs_dir):
    """Run the full pipeline for a video file, skipping cached steps."""
    config = load_config(config_path)
    jobs_root = Path(jobs_dir)

    job = init_step.run(video_file, video_id=video_id, jobs_root=jobs_root, force=force, echo=click.echo)
    resolved_id = job.video_id

    transcribe_step.run(resolved_id, config, jobs_root=jobs_root, force=force, echo=click.echo)
    translate_step.run(resolved_id, config, lang, jobs_root=jobs_root, force=force, echo=click.echo)
    subtitles_step.run(resolved_id, lang, jobs_root=jobs_root, force=force, echo=click.echo)
    tts_step.run(resolved_id, config, lang, jobs_root=jobs_root, force=force, echo=click.echo)
    output_path = mux_step.run(
        resolved_id,
        lang,
        jobs_root=jobs_root,
        hardsub=hardsub,
        keep_original_audio=keep_original_audio,
        force=force,
        echo=click.echo,
    )
    click.echo(f"pipeline complete: {output_path}")
