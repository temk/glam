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
- cache each translated segment to disk so a crash can be resumed without re-translating;
- preserve the segment structure from the transcript;
- create the `translation.<target>.json` artifact.

## CLI

```bash
uv run glam translate --job-id JOB_ID [--target LANG] [--config PATH] [--batch-size N] [--context-size N] [--start N] [--force]
```

- `--target` — target language code, overriding the job's default target from `job.yaml`.
- `--batch-size` — number of segments translated per model request (default: 100).
- `--context-size` — number of already-translated preceding segments sent read-only with each
  batch for continuity (default: 100).
- `--start` — resume from the N-th segment (1-based); earlier segments are taken from the cache
  (see "Caching and resume").

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

## Caching and resume

Translating a long transcript makes many model calls, so the step must not hold results only in
memory: a failure partway through would waste the work already done. Each translated segment is
written to a per-job cache directory as its own JSON file (`{id, translated_text}`):

```text
<job_dir>/<job-id>/translate/<target>.<segment-id>.json
```

The cache is keyed by target language (and the zero-padded segment id), so several languages coexist
in one job without clashing.

On each run the step first fills in every segment already present in the cache, then translates only
the remaining segments (batched, with the usual `context` from preceding translations — which may be
cached ones). This makes a rerun resume automatically: only the missing segments are sent to the
model. `--force` re-translates every segment, ignoring the cache.

`--start N` begins at the N-th segment (1-based, matching the progress counter). Segments before N are
taken from the cache and are **not** re-translated; if a required earlier segment is not cached, the
step fails with a clear error. Combine `--start N --force` to redo the tail from N while keeping the
earlier cached segments.

Unlike a plain run, passing `--force` or `--start` runs even when `translation.<target>.json` already
exists.

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

`translate` owns its final artifact and its segment cache:

```text
translation.<target>.json
translate/               # per-segment translation cache (see "Caching and resume")
```

It does not modify artifacts owned by other steps.

## Idempotency and `--force`

`translate` must be idempotent. A plain run whose `translation.<target>.json` already exists is
skipped; `--force` recomputes it and `--start` resumes it (both run even when the output exists). The
per-segment cache lets a rerun continue where a crash stopped instead of starting over.

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
- the model did not translate all segments (missing segment IDs) after all retry rounds;
- `--start` refers to a position whose earlier segments are not cached;
- unable to write the translation or cache a segment.

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
- caching each translated segment under `translate/`;
- resuming from the cache without re-requesting (e.g. after the final file is lost);
- re-translating every segment with `--force`;
- resuming translation from `--start N`, taking earlier segments from the cache;
- erroring when `--start` needs an earlier segment that is not cached.
