# Step: `accent`

`accent` applies per-language text fixes to the translated segments so the `tts` step can dub them
correctly. Its current job is Russian stress marks: TTS voices frequently misplace stress on Russian
words, so `accent` marks the stressed vowel in every segment.

The step is **optional** and **language-specific**. It only produces output for languages that have a
registered fixer; for any other target language it does nothing.

This step runs **locally**. For Russian it loads a bundled stress-prediction model
(`ruaccent-predictor`) on the machine running the CLI; it makes no network calls and uses no service
from the config.

## Purpose

`accent` must:

- read an existing job;
- read `job.yaml`;
- resolve the target language (`--target`, else `job.yaml`);
- if the target has no registered fixer, do nothing and report it;
- otherwise read `translation.<target>.json`;
- apply the target language's fixer to the `translated_text` of each segment;
- write the result as `translation.<target>.fixed.json`, preserving every other field.

## CLI

```bash
uv run glam accent --job-id JOB_ID [--target LANG] [--config PATH] [--force]
```

The CLI is defined in `src/glam/cli.py`.

The step module must remain independent of the CLI.

`--target` overrides the job's default target language from `job.yaml`.

## Input data

Expected artifacts:

```text
<job_dir>/<job-id>/
  job.yaml
  translation.<target>.json
```

## Output

For a language with a fixer, the step writes:

```text
<job_dir>/<job-id>/translation.<target>.fixed.json
```

The file has the same structure as `translation.<target>.json`; only the `translated_text` of each
segment is modified. Every other field (segment `id`/`start`/`end`, the source `text`, and all
top-level fields) is copied unchanged.

For a language without a fixer, the step writes nothing.

## Language fixers

The step keeps a registry mapping a target language to its text fixer. A language absent from the
registry is left untouched (no output).

- `ru` — Russian stress marks. Each segment is run through `ruaccent-predictor`. The stressed vowel
  is marked with the combining acute accent (`U+0301`), for example `Наде́юсь`, which is the stress
  form TTS engines expect.

Adding support for another language means registering a fixer for it; no other step changes.

## Consumption by `tts`

When `translation.<target>.fixed.json` exists, `tts` reads it in preference to
`translation.<target>.json`. `subtitles` always reads the plain `translation.<target>.json`, so
subtitles never contain stress marks.

## Idempotency and `--force`

`accent` is idempotent.

If `translation.<target>.fixed.json` already exists, the step skips work unless `--force` is given.
`--force` recomputes it from `translation.<target>.json`.

## Errors

Expected errors:

- job not found;
- missing `job.yaml`;
- invalid `job.yaml` format;
- missing target language (no `--target` and none in `job.yaml`);
- missing `translation.<target>.json`;
- invalid `translation.<target>.json` format.

Expected errors must be converted into clear CLI errors through the project's base error class.

## Tests

Tests for `accent` must cover:

- writing `translation.<target>.fixed.json` for a language with a fixer;
- applying the fixer to `translated_text` and preserving all other fields;
- doing nothing for a language without a fixer;
- skipping when the fixed artifact already exists;
- recomputing with `--force`;
- error when `translation.<target>.json` is missing;
- error when the job is not found.
