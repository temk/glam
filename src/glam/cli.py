import click
import functools
from importlib.metadata import version as pkg_version

from glam.common.config import DEFAULT_CONFIG_PATH, read_config
from glam.common.errors import GlamError


def handle_glam_errors(fn):
    """Report GlamError as a clean CLI message instead of a Python traceback."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except GlamError as e:
            raise click.ClickException(str(e)) from e

    return wrapper


config_option = click.option(
    "-c",
    "--config",
    "config_path",
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to config file",
)

target_option = click.option(
    "--target", "target", default=None, help="Target language code, overriding the job's target from job.yaml"
)

job_id_option = click.option("--job-id", "job_id", required=True, help="Id of the job to operate on")


@click.group()
def main():
    """GLAM — Glossary-Locked Audio Muxer."""


@main.command("version")
def version_cmd():
    """Print the glam version."""
    click.echo(pkg_version("glam"))


@main.command("init")
@click.argument("video_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--source", "source_lang", default=None, help="Source/original language code, e.g. en (default: defaults.source)"
)
@click.option("--target", "target_lang", default=None, help="Target language code, e.g. ru (default: defaults.target)")
@click.option(
    "--glossary",
    "glossary_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSON glossary to copy into the job",
)
@click.option("--voice", default=None, help="Default TTS voice for this job, stored in job.yaml")
@click.option("--job-id", "job_id", default=None, help="Job id (default: derived from filename)")
@config_option
@click.option("--force", is_flag=True, help="Recompute even if outputs already exist")
@handle_glam_errors
def init_cmd(video_file, source_lang, target_lang, glossary_path, voice, job_id, config_path, force):
    """Register a job from a local video file."""
    # Imported here, not at module top, so running one command does not import
    # every step and its backends. See docs/architecture.md "CLI layout".
    from glam.steps import init as init_step

    config = read_config(config_path)
    source_lang = source_lang or config.defaults.source
    target_lang = target_lang or config.defaults.target
    if not source_lang or not target_lang:
        raise click.UsageError(
            "missing language: pass --source and --target, or set defaults.source/target in the config"
        )
    job_path = init_step.run(
        video_file,
        source_lang=source_lang,
        target_lang=target_lang,
        glossary_path=glossary_path,
        voice=voice,
        job_id=job_id,
        jobs_root=config.job_dir,
        force=force,
        echo=click.echo,
    )
    click.echo(f"job '{job_path.name}' ready at {job_path}")


@main.command("transcribe")
@job_id_option
@config_option
@click.option("--force", is_flag=True, help="Recompute even if the transcript already exists")
@handle_glam_errors
def transcribe_cmd(job_id, config_path, force):
    """Transcribe a job's audio through the configured ASR service."""
    # Imported here, not at module top, so `--help` and local commands do not pull in
    # the OpenAI SDK. See docs/architecture.md "CLI layout".
    from glam.steps import transcribe as transcribe_step

    config = read_config(config_path)
    transcribe_step.run(job_id, config, force=force, echo=click.echo)


@main.command("translate")
@job_id_option
@target_option
@config_option
@click.option("--batch-size", type=int, default=None, help="Segments translated per request (default: 100)")
@click.option(
    "--context-size", type=int, default=None, help="Preceding translated segments sent for context (default: 100)"
)
@click.option(
    "--dump", is_flag=True, default=False, help="Dump each batch's request/response into translate.<lang>.dump/"
)
@click.option("--force", is_flag=True, help="Recompute even if the translation already exists")
@handle_glam_errors
def translate_cmd(job_id, target, config_path, batch_size, context_size, dump, force):
    """Translate a job's transcript through the configured LLM service."""
    # Imported here, not at module top, so `--help` and local commands do not pull in
    # the OpenAI SDK. See docs/architecture.md "CLI layout".
    from glam.steps import translate as translate_step

    # Only forward the tuning options when set, so the step keeps its own defaults.
    tuning = {k: v for k, v in {"batch_size": batch_size, "context_size": context_size}.items() if v is not None}
    config = read_config(config_path)
    translate_step.run(job_id, config, target=target, force=force, echo=click.echo, dump=dump, **tuning)


@main.command("subtitles")
@job_id_option
@target_option
@config_option
@click.option("--force", is_flag=True, help="Recompute even if the subtitles already exist")
@handle_glam_errors
def subtitles_cmd(job_id, target, config_path, force):
    """Render a job's translated segments into an SRT subtitle file."""
    # Imported here, not at module top, so running one command does not import every step.
    # See docs/architecture.md "CLI layout".
    from glam.steps import subtitles as subtitles_step

    config = read_config(config_path)
    subtitles_step.run(job_id, config, target=target, force=force, echo=click.echo)


@main.command("tts")
@job_id_option
@target_option
@click.option("--voice", default=None, help="Voice to synthesize with, overriding the job's voice from job.yaml")
@config_option
@click.option("--force", is_flag=True, help="Recompute even if the audio track already exists")
@handle_glam_errors
def tts_cmd(job_id, target, voice, config_path, force):
    """Synthesize a dubbed target-language audio track through the configured TTS service."""
    # Imported here, not at module top, so `--help` and local commands do not pull in
    # the OpenAI SDK. See docs/architecture.md "CLI layout".
    from glam.steps import tts as tts_step

    config = read_config(config_path)
    tts_step.run(job_id, config, target=target, voice=voice, force=force, echo=click.echo)


@main.command("mux")
@job_id_option
@click.option("--exclude", "exclude", multiple=True, help="Artifact to exclude (tts*.wav / subtitles*.srt); repeatable")
@config_option
@click.option("--force", is_flag=True, help="Rebuild even if the result already exists")
@handle_glam_errors
def mux_cmd(job_id, exclude, config_path, force):
    """Build the final MP4 from the source video, TTS tracks, and subtitles."""
    # Imported here, not at module top, so running one command does not import every step.
    # See docs/architecture.md "CLI layout".
    from glam.steps import mux as mux_step

    config = read_config(config_path)
    mux_step.run(job_id, config, exclude=exclude, force=force, echo=click.echo)
