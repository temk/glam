# Step: `translate`

`translate` is the third step in the GLAM pipeline. It takes the transcript created by the `transcribe` step, translates the segment text into the job's target language, and saves the result as a new job artifact.

The step runs local pipeline code, but the translation model is called through a remote OpenAI-compatible backend.

## Purpose

`translate` must:

- read an existing job;
- read `job.yaml`;
- get the source and target languages from `job.yaml`;
- find the `transcript.json` artifact;
- find the job-local glossary artifact `glossary.json`;
- call the translation service from the config;
- translate the text of each segment into the target language;
- apply glossary rules during translation;
- preserve the segment structure from the transcript;
- create the `translation.<target>.json` artifact.

## CLI

```bash
uv run glam translate --job-id JOB_ID [--target LANG] [--config PATH] [--batch-size N] [--context-size N] [--dump] [--force]
```

- `--target` — target language code, overriding the job's default target from `job.yaml`.
- `--batch-size` — number of segments translated per model request (default: 100).
- `--context-size` — number of already-translated preceding segments sent read-only with each
  batch for continuity (default: 100).
- `--dump` — write the raw model request/response exchange to per-batch dump files (see "Debug dump").

The CLI is defined in `src/glam/cli.py`.

The step module must remain CLI-agnostic.

## Input data

Expected artifacts:

```text
<job_dir>/<job-id>/
  job.yaml
  transcript.json
  glossary.json
```

`translate` reads the source and target languages from `job.yaml`.

## Config

The step uses a service named `translate`. Its `protocol` selects the backend (see docs/architecture.md "Service protocols" and "Backends"); `translate` supports `openai`, which uses `url` plus `params.model` and an optional `params.api_key`.

## Output

The step creates a per-language translation artifact:

```text
<job_dir>/<job-id>/translation.<target>.json
```

The target language is in the file name (for example `translation.ru.json`), so translations into
several languages coexist in one job. `translation.<target>.json` extends `transcript.json` by
adding the `translated_text` field to each segment.

The step prints per-batch progress with a timestamp while running.

## Debug dump

When `--dump` is set, the step writes the raw model exchange into a per-language dump folder in the
job directory, one file per batch:

```text
<job_dir>/<job-id>/translate.<target>.dump/<nnn>.json
```

`<target>` is the target language and `<nnn>` is the 1-based batch number, zero-padded to three digits so the files sort naturally. Each file is a JSON array of that batch's request/response
exchanges — every model call, including retry rounds and failed calls — recording the request
messages and the raw response content (or error). The file is rewritten after each request, so the
dump survives a mid-step crash. The folder is created at the start of a dumping run and cleared of
stale batch files from earlier runs.

Dump files are debug artifacts, not pipeline inputs: no downstream step reads them.

## Segment preservation

`translate` must preserve the segmentation from `transcript.json`.

The number of segments in `translation.<target>.json` must match the number of segments in `transcript.json`.

The segment order must match the segment order in `transcript.json`.

Timestamps must be copied without changes.

## Translation

The model prompt must explicitly instruct the model to follow the glossary wherever applicable.

If `glossary.json` is empty, translation is performed normally.

If `glossary.json` is missing or has an invalid format, this is an input error for the current step.

## Model request behavior

The step translates segments in batches of `--batch-size` (default 100).

Each batch request carries a JSON object with two fields:

- `translate` — the segments to translate, as an array of `{id, text}`;
- `context` — the already-translated target-language text of up to `--context-size` (default 100)
  preceding segments, as a single plain-text string (the joined `translated_text`, without ids).
  It is sent read-only so the model keeps terminology and wording consistent across batch
  boundaries. The model must not translate or return the context.

The request asks the model for a structured response that can be validated. The step uses
OpenAI-compatible structured outputs (a strict JSON schema), so backends that honor it (Ollama,
vLLM, OpenAI) are constrained to well-formed JSON; weak local models otherwise emit malformed JSON.

The prompt must instruct the model to translate every requested segment: exactly one entry per id,
the same number of entries as the input, without skipping, merging, splitting, reordering, adding,
or dropping segments.

The step must validate every response before writing `translation.<target>.json`:

- the response is well-formed and matches the expected shape;
- returned ids are a subset of the requested ids (unknown ids are an error);
- there are no duplicate ids.

Weak models sometimes drop a few segments from a large batch. The step re-requests only the missing
ids, up to a fixed number of rounds. If segments are still missing after the final round, this is an
error suggesting a smaller `--batch-size`.

Translation quality depends on the configured model; small models may still produce imperfect text
that a schema cannot fix.

## Artifact ownership

`translate` owns only its own artifact:

```text
translation.<target>.json
```

It does not modify artifacts owned by other steps.

## Idempotency and `--force`

`translate` must be idempotent.

## Errors

Expected errors:

- job not found;
- missing `job.yaml`;
- missing `transcript.json`;
- missing `glossary.json`;
- invalid `job.yaml`;
- missing source or target language in `job.yaml`;
- invalid `transcript.json` format;
- invalid `glossary.json` format;
- missing `translate` service in the config;
- translation service unavailable;
- translation service returned an invalid response;
- translation response was truncated by the model's output token limit;
- the model did not translate all segments (missing segment IDs) after all retry rounds.

Expected errors must be converted into clear CLI errors through the base error class.

## Tests

Tests for `translate` must cover:

- reading an existing job;
- reading the source and target languages from `job.yaml`;
- finding `transcript.json`;
- finding `glossary.json`;
- reading the `translate` service from the config;
- creating `translation.<target>.json`;
- skipping if `translation.<target>.json` already exists;
- recreating `translation.<target>.json` with `--force`;
- preserving segment IDs;
- preserving segment timestamps;
- preserving segment order;
- preserving the number of segments;
- writing translated text to `translated_text`;
- presence of `translated_text` in each segment of `translation.<target>.json`;
- after removing the `translated_text` fields, `translation.<target>.json` and `transcript.json` are identical;
- applying glossary entries in the model prompt;
- handling an empty glossary;
- handling an invalid glossary format;
- handling missing input artifacts;
- handling an invalid transcript format;
- handling an unavailable backend;
- handling an invalid backend response;
- handling a backend response with missing, duplicate, or unknown segment IDs;
- translating in batches of `--batch-size`;
- sending preceding translations as `context` for continuity;
- recovering segments the model drops from a batch by re-requesting them;
- printing per-batch progress;
- writing per-batch dump files incrementally when `--dump` is set;
- not writing dump files by default.
