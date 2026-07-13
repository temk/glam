# Step: `transcribe`

`transcribe` is the ASR step in the GLAM pipeline. It takes the prepared `audio.wav` from the job directory and creates a transcript with segment-level timestamps.

The step runs local pipeline code, but the ASR model is called through a remote OpenAI-compatible backend.

## Purpose

`transcribe` must:

- read an existing job;
- find `audio.wav`;
- call the ASR service from the config;
- save the raw ASR result in a normalized JSON format;
- heal the raw result into a cleaned transcript and record what it changed;
- merge the cleaned fragments into sentence-level segments so a sentence is not split across ids;
- save segment-level timestamps as a required part of the result.

## CLI

```bash
uv run glam transcribe --job-id JOB_ID [--config PATH] [--strict] [--force]
```

- `--strict` — also remove filler interjections while cleaning (see "Cleaning (healing) the
  transcript"). Off by default.

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

The step creates three artifacts:

```text
transcript.raw.json      # the untouched ASR result, kept for inspection
transcript.json          # the cleaned transcript downstream steps consume
transcript.cleanup.json  # the warnings produced while cleaning
```

`transcript.json` keeps its name so downstream steps (`translate`) read the cleaned transcript without
changing.

## Transcript format

`transcript.raw.json` and `transcript.json` share the same top-level shape. The only difference is
that each `transcript.json` segment additionally carries `source_ids` — the raw segment ids merged
into it (see "Merging into sentence units"). Minimal shape:

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
      "text": "Welcome to this lecture.",
      "source_ids": [0, 1]
    }
  ]
}
```

`source_ids` is a debug field owned by `transcribe`; it does not propagate downstream (`translate`
reads the transcript but drops it).

## Cleaning (healing) the transcript

Raw ASR output carries recurring artifacts that hurt translation, subtitles, and TTS. After saving
`transcript.raw.json`, the step heals it in two passes: it first cleans the raw segments with the rules
below, then merges the survivors into sentence units (see "Merging into sentence units"). The result
is written to `transcript.json`.

The cleaning rules are applied in order:

1. **Drop punctuation-only segments** — text with no letters or digits (for example `"."`, `"..."`, `"?!"`)
   is removed.
2. **Remove filler interjections** (only with `--strict`) — hesitation sounds carrying no meaning
   (English `uh`, `um`, `er`, `hmm`, `mm`, ...; Russian `э`, `эм`, `ммм`, `хм`, `хмм`, `а-а`, ...) are
   removed. Matching ignores case and surrounding punctuation, so `"э.."` and `"Хммм"` match. Fillers
   are removed both ways: a segment that is **only** fillers is dropped, and fillers **embedded** in a
   longer segment are cut from its text (`"Так, э, давайте начнём"` → `"Так, давайте начнём"`) while its
   timestamps are left unchanged. Without `--strict` this rule does nothing.
3. **Check `end > start`** — detection only: a segment whose `end` is not after its `start` is kept but a
   warning is recorded.
4. **Impossible speech rate** — detection only: when `len(text) / (end - start)` exceeds **25 chars/sec**
   (only computed when `end > start`) the segment is kept but a warning is recorded.
5. **Collapse consecutive identical segments** — neighbouring segments with equal text are merged into the
   first, whose `end` is extended to cover the duplicates.
6. **Collapse repeat-then-continuation** — when a segment's text repeats the previous segment's full text as a
   whole-word prefix and adds more (a common Whisper tail artifact), the shorter earlier segment is dropped and
   the survivor's `start` is pulled back to cover it.
7. **Renumber IDs** — surviving segments are renumbered sequentially from `0`.

Every removal or collapse also records a warning, so nothing is dropped silently. The filler rule
records `filler_only` when it drops a whole segment and `filler_removed` when it strips fillers from
within one.

### `transcript.cleanup.json`

The warnings are written as a separate report. Each warning references the segment `id` from
`transcript.raw.json` (before renumbering):

```json
{
  "version": 1,
  "step": "transcribe",
  "job_id": "example-video",
  "warnings": [
    { "rule": "punctuation_only", "segment_id": 7, "text": ".", "detail": "removed punctuation-only segment" }
  ]
}
```

## Merging into sentence units

ASR cuts on pauses, not grammar, so a single sentence is often split across several fragments. That
splitting makes the downstream translator merge or drop segments and lose the id alignment. After
cleaning, the step therefore merges adjacent segments into sentence-level units.

Two consecutive segments are merged when none of the **stop** conditions hold and at least one
**continue** condition does:

- **stop** (never merge): the gap between them is longer than `1.2 s`, or the earlier segment ends
  with `?` or `!`. (Speaker change would also stop a merge, but the ASR result carries no speaker
  information, so it is not yet applied.)
- **continue** (merge): the gap is shorter than `0.8 s` and the earlier segment does not end a
  sentence (`.`, `?`, `!`); or the earlier segment ends with a continuation cue (`and`, `or`, `but`,
  `because`, `which`, `that`, `so`, `uh`, `um`, a comma, or a colon); or the next segment starts with
  a lowercase letter.

A growing unit has size limits, applied so a unit is not cut in the middle of a sentence:

- **soft target** (**12 s**, **250 characters**, or **5 source segments**) — once a unit reaches one
  of these it is closed, but only when it already ends on a sentence boundary (`.`, `?`, `!`). While
  it ends mid-sentence the unit keeps merging toward the next boundary.
- **absolute ceiling** (**20 s**, **400 characters**, or **8 source segments**) — a hard stop that
  closes the unit even mid-sentence, so a run without any sentence boundary cannot grow without bound.

Each resulting unit gets a new sequential `id` (renumbered from `0`) and records `source_ids`, the
list of raw segment ids it absorbed, for debugging.

## Idempotency and `--force`

The two transcripts are produced lazily and independently:

- `transcript.raw.json` triggers a fresh ASR call only when it is missing; when it already exists it is
  reused as-is (no network call).
- `transcript.json` is re-cleaned from the raw transcript only when it is missing.

`--force` regenerates both (a new ASR call and a fresh clean). A useful consequence: deleting only
`transcript.json` re-runs the cleaning rules against the cached `transcript.raw.json` without re-transcribing.

## Artifact ownership

`transcribe` owns only its own artifacts:

```text
transcript.raw.json
transcript.json
transcript.cleanup.json
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
- creating the `transcript.raw.json`, `transcript.json`, and `transcript.cleanup.json` artifacts;
- healing the raw segments (each cleaning rule and the resulting warnings);
- skipping when `transcript.json` already exists;
- recreating with `--force`;
- handling segment-level timestamps;
- errors for missing input and invalid backend response.
