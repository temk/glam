# Step: `subtitles`

`subtitles` is the fourth step in the GLAM pipeline. It takes the translated segments from `translation.<target>.json` and creates a subtitle file for the target language.

This step runs **locally**.

## Purpose

`subtitles` must:

- read an existing job;
- read `job.yaml`;
- get the target language from `job.yaml`;
- find the `translation.<target>.json` artifact;
- validate the structure of the translated segments;
- create a subtitle file from `translated_text` and the segment timestamps;
- save the subtitles as an artifact of the current job.

## CLI

```bash
uv run glam subtitles --job-id JOB_ID [--target LANG] [--config PATH] [--force]
```

`--target` overrides the job's default target language from `job.yaml`; the step reads
`translation.<target>.json` and writes `subtitles.<target>.srt` for that language.

The CLI is defined in `src/glam/cli.py`.

The step module must remain CLI-agnostic.

## Output

The step creates a subtitle file:

```text
<job_dir>/<job-id>/subtitles.<target>.srt
```

The subtitle format is SRT.

## Segment validation

Before writing the file, the step must validate that each segment contains:

- `id`;
- `start`;
- `end`;
- `translated_text`.

`start` and `end` must be numbers.

`end` must be greater than `start`.

The segment order must match the order in `translation.<target>.json`.

## Artifact ownership

`subtitles` owns only its own artifact:

```text
subtitles.<target>.srt
```

It does not modify artifacts owned by other steps.

## Idempotency and `--force`

`subtitles` must be idempotent.

If the subtitle file already exists, the step must not recreate it without `--force`.

## Errors

Expected errors:

- job not found;
- missing `job.yaml`;
- missing `translation.<target>.json`;
- invalid `job.yaml`;
- missing target language in `job.yaml`;
- invalid `translation.<target>.json` format;
- missing `segments` list;
- segment does not contain `start`, `end`, or `translated_text`;
- `start` or `end` has an invalid type;
- `end` is less than or equal to `start`;
- unable to write the subtitle file.

Expected errors must be converted into clear CLI errors through the project's base error class.

## Tests

Tests for `subtitles` must cover:

- reading an existing job;
- reading the target language from `job.yaml`;
- finding `translation.<target>.json`;
- creating `subtitles.<target>.srt`;
- correct numbering of subtitle cues;
- correct formatting of SRT timestamps;
- using `translated_text`;
- preserving segment order;
- splitting long text into readable lines;
- skipping the step if the file already exists;
- recreating the file with `--force`;
- error for missing `translation.<target>.json`;
- error for invalid `translation.<target>.json` format;
- error for missing `translated_text`;
- error for invalid timestamps.
