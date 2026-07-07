# Step: `transcribe`

`transcribe` is the ASR step in the GLAM pipeline. It takes the prepared `audio.wav` from the job directory and creates a transcript with segment-level timestamps.

The step runs local pipeline code, but the ASR model is called through a remote OpenAI-compatible backend.

## Purpose

`transcribe` must:

- read an existing job;
- find `audio.wav`;
- call the ASR service from the config;
- save the transcript in a normalized JSON format;
- save segment-level timestamps as a required part of the result.

## CLI

```bash
uv run glam transcribe <job-id> [--config PATH] [--force]
```

The CLI is defined in `src/glam/cli.py`.

The step module must remain CLI-agnostic.

## Inputs

Expected artifacts:

```text
<job_dir>/<job-id>/
  job.yaml
  audio.wav
```

The source language is read from `job.yaml`.

## Config

The step uses the service named `transcribe`. Its `protocol` selects the backend (see docs/architecture.md "Service protocols" and "Backends"); `transcribe` supports `openai`, which uses `url` plus `params.model` and an optional `params.api_key`.

The step must request a response format with segment-level timestamps.

## Output

The step creates a transcript artifact:

```text
transcript.json
```

## Transcript format

Minimal format:

```json
{
  "version": 1,
  "step": "transcribe",
  "job_id": "example-video",
  "source_language": "en",
  "model": "Systran/faster-whisper-large-v3",
  "audio_artifact": "audio.wav",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Welcome to this lecture."
    }
  ]
}
```

## Artifact ownership

`transcribe` owns only its own artifacts:

```text
transcript.json
```

It does not modify artifacts owned by other steps.

## Errors

Expected errors:

- job not found;
- missing `job.yaml`;
- missing `audio.wav`;
- no `transcribe` service in the config;
- ASR service is unavailable;
- ASR response is invalid;
- response does not contain segment-level timestamps.

## Tests

Tests should cover:

- reading an existing job;
- finding `audio.wav`;
- reading source language from `job.yaml`;
- reading the `transcribe` service from the config;
- creating the transcript artifact;
- skipping when the artifact already exists;
- recreating with `--force`;
- handling segment-level timestamps;
- errors for missing input and invalid backend response.
