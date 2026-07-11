# GLAM — General Architecture

This document describes only the high-level architecture of GLAM: the pipeline step order, shared execution parameters, the general configuration format, and the boundary between local execution and remote model-backed services.

Details for specific steps should live in separate files:

- `docs/steps/init.md`
- `docs/steps/transcribe.md`
- `docs/steps/translate.md`
- `docs/steps/subtitles.md`
- `docs/steps/accent.md`
- `docs/steps/tts.md`
- `docs/steps/mux.md`
- `docs/steps/run.md`

## Pipeline steps

Main execution order:

```text
init → transcribe → translate → subtitles → accent → tts → mux
```

`accent` is optional: it corrects the translated text for languages that need it (currently Russian
stress marks) and only produces output for those languages. `subtitles` and `accent` both consume
`translate`'s output and are independent of each other.

`run` is an orchestration command that executes these steps in sequence and skips artifacts that already exist.

`run` must not implement the internal logic of individual steps. It only determines execution order, passes shared parameters, invokes steps in the correct sequence, and respects idempotency/`--force`.

Step roles in brief:

| Step | Purpose |
|---|---|
| `init` | Registers a job, creates the job directory, writes the job manifest and glossary, and prepares initial local artifacts. |
| `transcribe` | Produces a transcript and timestamps through an ASR backend. |
| `translate` | Translates the transcript through an LLM backend while applying glossary rules. |
| `subtitles` | Creates a subtitle file from translated segments. |
| `accent` | Applies per-language text fixes (e.g. Russian stress marks) into `translation.<target>.fixed.json`. |
| `tts` | Creates a dubbed audio track through a TTS backend. |
| `mux` | Builds the final video container from the source video, subtitles, and audio tracks. |
| `run` | Runs the full pipeline end-to-end. |

## CLI layout

All CLI wiring lives in `src/glam/cli.py`: the root command group, every subcommand, and each subcommand's arguments and options. `click` is used only here — step modules must not import `click` or any other CLI framework.

Each subcommand is a thin layer:

- its flags and options are declared in `cli.py`. Click must know a command's options before it parses the arguments, so they cannot come from a module that is imported lazily;
- its body imports the corresponding step module lazily, inside the function, and delegates the actual work to that module;
- reading the config, formatting output, and error handling stay in `cli.py`.

```python
@main.command("init")
@click.argument("video_file", ...)
@config_option
@handle_glam_errors
def init_cmd(video_file, config_path, ...):
    from glam.steps import init as init_step   # imported only when `init` runs
    config = read_config(config_path)
    job = init_step.run(..., echo=click.echo)
    click.echo(...)
```

Because the step module is imported inside the command body, running one command imports only that step and its dependencies — not every step. This keeps `glam --help`, `glam version`, and any single command from pulling in unrelated backends (for example the OpenAI SDK used by remote steps).

Step modules expose plain, CLI-agnostic functions — for example `run(...)` taking an `echo` callable that defaults to `print`. Step logic stays free of the CLI framework and easy to test in isolation.

Shared CLI helpers — the `--config` option and the `GlamError`-to-CLI error handler — live in `cli.py` and are applied to each command as decorators.

## Shared step parameters

All steps should rely on two shared concepts:

### Config file

Configuration is loaded from a config file.

Default path:

```text
~/.glam.yaml
```

The CLI may allow overriding the config file path explicitly, for example with `--config` / `-c`.

If the config file path is passed explicitly through the CLI, it takes precedence over the default path.
The default path is used only when no explicit path is provided.

### Job ID

`job-id` identifies a specific video processing job and links all pipeline artifacts together.

All steps after `init` should operate on an existing `job-id` and look for input/output files inside the corresponding job directory.

### Job manifest

Each job directory contains a `job.yaml` manifest created by `init`.

`job.yaml` is the source of truth for metadata and run-level parameters that belong to this specific video job.

Minimal structure:

```yaml
version: 1

job:
  id: example-video
  created_at: "2026-07-05T12:34:56+03:00"

source:
  original_path: "/original/path/video.mp4"
  filename: "video.mp4"
  artifact: "source.mp4"
  audio_artifact: "audio.wav"
  duration_seconds: 123.45

languages:
  source: en
  target: ru

# optional default TTS voice; omitted (null) when not set at init
voice: nova
```

`languages.source` and `languages.target` are job-level parameters. They are provided during `init` and are later reused by downstream steps.

`languages.target` is the job's default target language. The `translate`, `subtitles`, `accent`, and `tts` steps accept `--target` to override it per run and write per-language artifacts (for example `translation.ru.json`, `subtitles.ru.srt`, `translation.ru.fixed.json`, `tts.ru.wav`), so one job can hold outputs for several languages. `voice` is the job's default TTS voice, which `tts` may override with `--voice`.

The `accent` step writes a corrected translation as `translation.<target>.fixed.json`. When it exists, `tts` reads it in preference to `translation.<target>.json`, so dubbing uses the fixed text while `subtitles` keeps reading the plain translation.

### Job glossary

Each job may contain a job-local glossary artifact:

```text
glossary.json
```

The glossary is stored as JSON because glossary keys may be full phrases or sentences.

When a glossary path is passed to `init`, the file is copied into the job directory as `glossary.json`. Downstream steps must read only the job-local `glossary.json`, not the original external path.

This allows the glossary to be edited for a single video without mutating a shared glossary file.

## Data flow between steps

Pipeline steps must not call each other directly.

Each step:

- reads input artifacts from the job directory;
- reads job metadata and job-level parameters from `job.yaml`;
- creates its own output artifacts in the job directory;
- does not keep pipeline state in memory between commands.

This allows steps to be run independently, an individual step to be repeated, the backend/config to be changed, and `run` to remain only an orchestration layer.

## Artifact names

Output artifact names should be deterministic and built from parameters that affect the result.

Usually this includes:

- `job-id`, through placement inside the job directory;
- the step name;
- the target language, when applicable;
- backend/model/voice/config variant, when it affects the result.

This is required for caching, reproducibility, and comparing results from different backend/model choices.

## Idempotency and `--force`

Each pipeline step must be idempotent.

If a step's output artifact already exists, the step must not recompute it.

`--force` explicitly allows recreating the output of the current step.

General rule:

```text
output exists + no --force → skip
output missing → run
output exists + --force → recompute
```

A step must not accidentally overwrite artifacts created by another backend/model/config variant. If different backends/models produce different results, this must be reflected in the artifact name or location.

A step whose work is expensive and made of many independent items may additionally cache those items on disk so a failure can be resumed without redoing finished work. Such a cache is an internal resume aid, not the step's output: reruns reuse it, `--force` rebuilds it, and the step may offer a resume option to start partway through. `translate` and `tts` do this — they cache each finished segment under `<job>/translate/` and `<job>/tts/` and take `--start N` to resume (see `docs/steps/translate.md` and `docs/steps/tts.md`).

## Local and remote steps

Part of the pipeline runs locally, where the CLI is invoked.

Local steps:

- `init`
- `subtitles`
- `accent`
- `mux`
- `run` as an orchestration layer

Local steps may use the filesystem, CPU, and local utilities such as `ffmpeg`, but they must not require a remote model backend. `accent` is a local step that may load a bundled local model (the Russian accentor runs on the CPU/GPU of the machine invoking the CLI); it has no configured service entry and makes no network calls.

Remote model-backed steps:

- `transcribe`
- `translate`
- `tts`

Remote steps must not assume a local GPU or a locally installed model runtime. They call an external service over HTTP.

## Service protocols

Every remote model-backed service declares a `protocol` in its config entry. The protocol selects how the service is called and which additional config fields that entry carries.

- `openai` — the OpenAI-compatible protocol. It is the common choice and works for any OpenAI-compatible backend: ASR in `transcribe`, LLM in `translate`, TTS in `tts`.
- `chatterbox` and other named protocols — native protocols for a specific server, used when it exposes capabilities the OpenAI request shape cannot express. For `tts`, Chatterbox's native `/tts` endpoint adds per-request `language` selection and voice cloning. Each native protocol must be documented in the owning step's doc.

Pipeline steps must not hard-code a provider or protocol. Selecting or replacing a backend is a configuration change (`protocol`, `url`, and protocol-specific fields), not a change to a step's business logic. Because the request shape and available fields depend on the protocol, a service entry's config schema depends on its `protocol` (see "Configuration structure").

## Backends

Remote steps call their model backend through a small per-step backend abstraction instead of building a client inline. The abstraction lives in the `glam.backend` package, one subpackage per step that has a remote backend:

```text
src/glam/backend/
  tts/         base interface + one module per protocol (openai, chatterbox, ...)
  transcribe/  base interface + openai
  translate/   base interface + openai
```

Each subpackage's `base.py` defines the step's backend interface (for example `TtsBackend.synthesize(...)`) and a factory that dispatches on the service `protocol`, lazily importing only the selected implementation so choosing one backend never imports another backend's SDK. Implementation modules (`openai.py`, `chatterbox.py`, ...) hold the protocol client and parse their own protocol-specific config from the service entry.

`glam.common` stays free of these clients and their heavy SDK imports: it parses only the common part of a service entry (`name`, `protocol`, `url`) and hands the rest to the backend, which validates the protocol-specific fields when the step builds it.

## Errors

Expected failures should be converted into clear CLI errors.

Examples of expected failures:

- missing config file;
- unknown `job-id`;
- missing input artifact from a previous step;
- multiple matching input/output artifacts found and the choice is ambiguous;
- remote service is unavailable;
- remote service returned a response with an invalid format.

Such errors should use the base error class from `glam.common.errors`.

## Shared code in `glam.common`

`glam.common` is the package of code shared by all pipeline steps. Its `__init__.py` stays empty; each shared concern lives in its own submodule:

- `common/errors.py` — the base error class for the project (`GlamError`);
- `common/config.py` — `dataclass Config`, its service definitions, and `read_config`, the function for reading the config file;
- `common/job.py` — the job manifest (`job.yaml`) dataclasses and its read/write helpers;
- `common/translation.py` — reading and validating the `translation.<target>.json` artifact shared by `subtitles` and `tts`, plus the per-target artifact names (`translation_filename`, and `fixed_translation_filename` for the `accent` step's `translation.<target>.fixed.json`).

The client for a remote service lives in that service's backend module under `glam.backend` (see "Backends"), not in `glam.common`: `glam.common` is imported by every step, including local ones, and must not pull in a heavy SDK import such as the OpenAI SDK.

Step-specific logic must not grow inside `glam.common`. Details of a specific step should stay in that step's module.

Types and classes used only in one module must be declared in that module. Only move a type or class into a shared module when at least two modules use it.

## Configuration structure

The config file should contain shared pipeline settings and service definitions.

Minimal structure:

```yaml
job_dir: ./jobs

defaults:
  source: en
  target: ru

services:
  - name: transcribe
    protocol: openai
    url: http://host:8000/v1
    params:
      model: some-asr-model

  - name: translate
    protocol: openai
    url: http://host:11434/v1
    params:
      model: some-llm-model
      api_key: SECRET

  - name: tts
    protocol: chatterbox
    url: http://host:8004
```

### `job_dir`

`job_dir` defines the base directory where job directories and pipeline artifacts are stored.

### `defaults`

`defaults` is an optional section holding job-level defaults. Currently it carries the default languages:

- `source` — default source language, used by `init` when `--source` is omitted;
- `target` — default target language, used by `init` when `--target` is omitted.

Both are optional. A per-run `--source`/`--target` flag always overrides the default, and if neither the flag nor the default is set, `init` reports a clear error. Job-level parameters (like the languages) are still recorded per job in `job.yaml`; `defaults` only seeds them at `init` time.

### `services`

`services` describes remote model-backed services.

Every entry has three **required** common fields plus one optional protocol-specific bag:

- `name` — the step or service name, for example `transcribe`, `translate`, `tts`;
- `protocol` — selects how the service is called (`openai`, `chatterbox`, ...) and therefore which fields `params` carries;
- `url` — base URL of the service;
- `params` — an optional mapping (default `{}`) of protocol-specific fields.

The shared config loader validates only the common part and keeps `params` as a plain mapping. Each backend deserializes `params` into its own typed config and validates it when the step builds the backend — so the schema of `params` depends on `protocol`:

- `openai` — `params.model` (model name); `params.api_key` (optional API key; real keys belong only in local, uncommitted config files); for `tts`, an optional default `params.voice` (the OpenAI speech protocol requires a voice, so this or a per-run `--voice` must be set). The `url` points at the OpenAI-compatible base (usually ending in `/v1`).
- `chatterbox` (`tts` only) — the `url` points at the server root; the native `/tts` endpoint is appended by the backend. `params` may be empty; the server requires a predefined voice, so when none is resolved the backend falls back to a built-in default voice.

Step-specific tuning does not belong to the config file; it lives in CLI options, job-local files such as `job.yaml` and `glossary.json`, and constants of the corresponding step module.

Source language, target language, and glossary path are not global config values. They are parameters of a specific job and are passed to `init` with `--source`, `--target`, and `--glossary`.
