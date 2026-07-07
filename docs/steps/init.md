# Step: `init`

`init` is the first step in the GLAM pipeline. It creates a job for a local video file and prepares the base artifacts required by the following steps.

This step runs **locally**. It does not use a remote model backend and does not call OpenAI-compatible services.

## Purpose

`init` must:

- accept a local video file;
- determine or accept a `job-id`;
- create the job directory;
- save the source video as a job artifact;
- extract basic video metadata;
- accept source and target languages for this job;
- create the job manifest;
- create a job-local glossary artifact;
- prepare an audio artifact for the later `transcribe` step.

## CLI

The CLI for `init` should be defined in `src/glam/cli.py`, like the other CLI commands. The `init` step module should expose CLI-agnostic logic.

Expected command shape:

```bash
uv run glam init <video_file> [--source SOURCE_LANG] [--target TARGET_LANG] [--glossary PATH] [--voice VOICE] [--job-id JOB_ID] [--config PATH] [--force]
```

`--source` and `--target` are required only when the config has no `defaults.source`/`defaults.target` to fall back on.

## Inputs

`init` accepts:

- `video_file` — path to a local video file;
- `--source` — source/original language code, for example `en`; optional when the config's `defaults.source` is set (the flag overrides the default);
- `--target` — target language code, for example `ru`; optional when the config's `defaults.target` is set (the flag overrides the default);
- `--glossary` — optional path to a JSON glossary file to copy into the job directory;
- `--voice` — optional default TTS voice for this job, stored in `job.yaml` and used later by `tts`;
- `config file` — path to the config file, defaulting to `~/.glam.yaml`;
- `job-id` — optional explicit job id;
- `--force` — explicit permission to recreate the artifacts of the current step.

If `job-id` is not provided explicitly, it must be generated and printed.

## Output artifacts

`init` creates a job directory inside `job_dir`.

Minimum expected artifacts:

```text
<job_dir>/<job-id>/
  job.yaml
  glossary.json
  source.<ext>
  audio.wav
```

### `job.yaml`

`job.yaml` is the job manifest. It must contain metadata needed by later steps and useful for human debugging.

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

# optional; omitted (null) when `--voice` is not passed
voice: nova
```

`target` is the job's default target language; downstream steps (`translate`, `subtitles`, `tts`)
may override it per run with `--target`. `voice` is the job's default TTS voice; `tts` may override
it per run with `--voice`.

### `glossary.json`

`glossary.json` is the job-local glossary artifact.

Regardless of the input form, the stored `glossary.json` must always be a JSON object mapping strings to strings (`term -> translation`). Keys and values are strings; keys may be full phrases or sentences.

If `--glossary PATH` is provided, `init` normalizes the input into that map based on the file:

- a `.json` file is parsed by content:
  - a JSON **object** is taken as the map as-is (its keys and values must be strings);
  - a JSON **array** of terms becomes an identity map, where each term maps to itself (`"term": "term"`);
- any other file is treated as **plain text**: each non-empty line is one term, stored as an identity map entry (`"term": "term"`).

A malformed input (invalid JSON in a `.json` file, a non-string entry, or a top-level JSON value that is neither object nor array) is an error reported through the base error class.

If `--glossary` is not provided, `init` creates an empty glossary file:

```json
{}
```

Downstream steps must use only the job-local `glossary.json`, not the original external glossary path.

### `source.<ext>`

The source video file must be saved as a job artifact.

### `audio.wav`

`audio.wav` is the prepared audio artifact for `transcribe`.

It must be created locally through `ffmpeg`.

Recommended format:

- WAV;
- mono;
- 16 kHz sample rate.

## Idempotency and `--force`

`init` must be idempotent.

`--force` must recreate the output of the current step.

## Artifact ownership

`init` owns only its own artifacts:

- `job.yaml`;
- `glossary.json`;
- `source.<ext>`;
- `audio.wav`.

## Local execution

`init` runs locally and may use:

- filesystem;
- local CPU;
- `ffmpeg`;
- `ffprobe`.

## Errors

Expected failures must be converted into clear CLI errors through the base error class from `common.py`.

Errors should explain:

- what exactly failed;
- which path or parameter caused the problem;
- what the user can do next, when that is obvious.

## Interaction with `common.py`

`init` must not add step-specific logic to `common.py`.

If logic is needed only for `init`, it should live in the `init` module or next to it.

## Tests

Tests for `init` should cover:

- job directory creation;
- `job-id` generation;
- explicit `job-id`;
- reading `job_dir` from config;
- writing `job.yaml`;
- copying a provided glossary to `glossary.json`;
- creating an empty `glossary.json` when no glossary is provided;
- storing an optional `--voice` in `job.yaml` (and leaving it unset otherwise);
- falling back to the config's `defaults.source`/`defaults.target` when the flags are omitted;
- erroring when a language is set neither by flag nor by `defaults`;
