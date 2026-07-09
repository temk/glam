# Step: `tts`

`tts` is the fifth step in the GLAM pipeline. It takes the translated segments from `translation.<target>.json` and creates a dubbed audio track in the target language.

This step executes local pipeline logic, but the speech synthesis model is called through a remote backend selected by the service `protocol` (an OpenAI-compatible backend or a native one such as Chatterbox).

## Purpose

`tts` must:

- read an existing job;
- read `job.yaml`;
- get the target language from `job.yaml`;
- find the `translation.<target>.json` artifact;
- validate the structure of translated segments;
- call the TTS service from the config;
- synthesize speech for the `translated_text` of each segment;
- cache each synthesized segment to disk so a crash can be resumed without re-synthesizing;
- combine the segments into a single audio track;
- save the audio track as an artifact of the current job.

## CLI

```bash
uv run glam tts --job-id JOB_ID [--target LANG] [--voice VOICE] [--start N] [--config PATH] [--force]
```

The CLI is defined in `src/glam/cli.py`.

The step module must remain independent of the CLI.

`--target` overrides the job's default target language from `job.yaml`; the step reads
`translation.<target>.json` for that language.

`--voice` selects the voice, overriding the job's default `voice` from `job.yaml`. Voice resolution
is: `--voice`, then the job's `voice`, otherwise it is left unset and the TTS backend picks its own
default. When a voice is set by either `--voice` or `job.yaml`, it is included in the artifact name.
Voices are configured without a file extension; the chatterbox backend appends `.wav` when it builds
the server's `predefined_voice_id` (so voice `Michael` becomes id `Michael.wav`).

## Input data

Expected artifacts:

```text
<job_dir>/<job-id>/
  job.yaml
  translation.<target>.json
```

`tts` reads the target language from `job.yaml`.

When the `accent` step's corrected artifact `translation.<target>.fixed.json` exists, `tts` reads it in preference to `translation.<target>.json`. This lets dubbing use language-specific fixes (such as Russian stress marks) while `subtitles` keeps reading the plain translation. Both files share the same structure.

The text for speech synthesis is taken only from the `translated_text` field of each segment in the chosen translation artifact.

## Config

The step uses the service named `tts`. Its `protocol` selects the backend and the extra config fields (see docs/architecture.md "Service protocols" and "Backends"). `tts` supports two protocols:

- `openai` — the OpenAI-compatible speech protocol. `url` is the base ending in `/v1`; `params` carries `model`, optional `api_key`, and an optional default `voice`. The OpenAI speech protocol **requires** a voice, so a voice must come from `--voice`, the job's `voice` in `job.yaml`, or `params.voice`; if none is set the step fails with a clear error.
- `chatterbox` — the native Chatterbox-TTS-Server protocol (`POST /tts`). `url` is the server root (the backend appends `/tts`); `params` may be empty. It sends the target language as the request `language`, so the multilingual server dubs into the target language. The server requires a `predefined_voice_id` (it has no implicit default), so the backend sends the resolved voice or, when none is set, falls back to a built-in default voice (`DEFAULT_VOICE`, one of the server's predefined voices). Generative parameters (`exaggeration`, `cfg_weight`, `temperature`, `seed`, ...) are left at the server's defaults.

Voice resolution order is `--voice`, then the job's `voice`, then (for `openai`) `params.voice`.

Using a native protocol here is the documented exception allowed by docs/architecture.md "Service protocols": the OpenAI speech shape cannot express per-request `language` selection or voice cloning, which the multilingual Chatterbox server provides through its native `/tts` endpoint.

## Output

The step creates an audio artifact:

```text
<job_dir>/<job-id>/tts.<target>.wav
```

The audio format is WAV.

## Caching and resume

Synthesizing every segment is slow, so the step must not hold all fragments only in memory: a failure
partway through would waste the work already done. Each synthesized segment is written to a per-job
cache directory as its own WAV file:

```text
<job_dir>/<job-id>/tts/<target>[.<voice>].<segment-id>.wav
```

The cache is keyed by target language and voice (and the zero-padded segment id), so several
languages and voices coexist in one job without clashing.

On each run the step processes segments in order and, for each one:

- if a cached file for it already exists and `--force` is not set, it is reused (no synthesis);
- otherwise the segment is synthesized and written to the cache.

This makes a rerun resume automatically: only the missing segments are synthesized. `--force`
re-synthesizes every segment, ignoring the cache.

`--start N` begins at the N-th segment (1-based, matching the progress counter). Segments before N are
taken from the cache and are **not** synthesized; if a required earlier segment is not cached, the
step fails with a clear error. Combine `--start N --force` to redo the tail from N while keeping the
earlier cached segments.

Unlike other steps, an existing final `tts.<target>.wav` is skipped only on a plain run; passing
`--force` or `--start` runs anyway.

## Synchronization

The basic version of the step must preserve the segment order and create a continuous audio track.

When possible, synthesized fragments should be placed according to the original segment timestamps:

- the fragment starts at `start`;
- if the fragment is shorter than the segment, the remaining time is filled with silence;
- if the fragment is longer than the segment, the step must not silently trim the audio without an explicit decision in the code.

The behavior for overly long fragments must be deterministic and covered by tests.

## Backend behavior

The step builds a TTS backend from the service `protocol` (see docs/architecture.md "Backends") and synthesizes each segment through it. The backend returns WAV audio for a segment's `translated_text`; the step is otherwise protocol-agnostic and performs the same validation and assembly regardless of protocol.

The step may send segments to the TTS backend one by one or in batches, if the backend supports it.

The backend response must be validated before writing the final file.

An invalid or incomplete backend response must be converted into a clear error.

## Artifact ownership

`tts` owns its final artifact and its segment cache:

```text
tts.<target>.wav
tts/                     # per-segment WAV cache (see "Caching and resume")
```

## Errors

Expected errors:

- job not found;
- missing `job.yaml`;
- missing `translation.<target>.json`;
- invalid `job.yaml`;
- missing target language in `job.yaml`;
- invalid `translation.<target>.json` format;
- missing `segments` list;
- a segment does not contain `start`, `end`, or `translated_text`;
- `start` or `end` has an invalid type;
- `end` is less than or equal to `start`;
- `translated_text` has an invalid type;
- missing `tts` service in the config;
- unknown or unsupported service `protocol`;
- no voice resolved for the `openai` protocol (which requires one);
- TTS service is unavailable;
- TTS service returned an invalid response;
- `--start` refers to a position whose earlier segments are not cached;
- unable to assemble the final audio track;
- unable to write the audio file or cache a segment.

Expected errors must be converted into clear CLI errors through the project's base error class.

## Tests

Tests for `tts` must cover:

- reading an existing job;
- reading the target language from `job.yaml`;
- finding `translation.<target>.json`;
- reading the `tts` service from the config;
- selecting the backend from the service `protocol`;
- creating `tts.<target>.wav`;
- including `--voice` in the artifact name if the voice affects the result;
- resolving the voice in `--voice` → job `voice` → service `voice` order;
- erroring when the `openai` protocol resolves no voice;
- sending the target language as `language` on the `chatterbox` protocol;
- using `translated_text`;
- preserving segment order;
- validating `start` and `end`;
- adding silence between segments when needed;
- handling an overly long synthesized fragment;
- skipping the step if the file already exists;
- caching each synthesized segment under `tts/`;
- resuming from the cache without re-synthesizing (e.g. after the final track is lost);
- re-synthesizing every segment with `--force`;
- resuming synthesis from `--start N`, taking earlier segments from the cache;
- erroring when `--start` needs an earlier segment that is not cached;
- recreating the file with `--force`;
- an error when `translation.<target>.json` is missing;
- an error when `translation.<target>.json` has an invalid format;
- an error when `translated_text` is missing;
- an error for invalid timestamps;
- an error when the `tts` service is missing from the config;
- an error when the backend is unavailable;
- an error when the backend response is invalid.
