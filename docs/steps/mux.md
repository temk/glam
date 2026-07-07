# Step: `mux`

`mux` is the final step in the GLAM pipeline. It builds the final MP4 from the source video and all discovered dubbing and subtitle artifacts.

This step runs **locally** and does not call a model backend.

## Purpose

`mux` must:

- read an existing job;
- read `job.yaml`;
- find the source video from `source.artifact`;
- find all artifacts matching `tts*.wav` and `subtitles*.srt` in the job directory;
- apply exclusions from `--exclude`;
- build the final `.mp4` through `ffmpeg`;
- create `result.mp4` as a softlink to the final file.

## CLI

```bash
uv run glam mux <job-id> [--exclude ARTIFACT]... [--config PATH] [--force]
```

`--exclude` may be passed multiple times. Its value is an artifact name inside the job directory, for example:

```bash
uv run glam mux example-video --exclude subtitles.en.srt --exclude tts.he.wav
```

Only discovered subtitle/audio artifacts of this step may be excluded: `subtitles*.srt` and `tts*.wav`. A non-existing artifact passed to `--exclude` is an error, so typos are not silently ignored.

The CLI is defined in `src/glam/cli.py`.

The step module must remain independent of the CLI.

## Input data

Minimum:

```text
<job_dir>/<job-id>/
  job.yaml
  source.<ext>
```

Additionally, the step picks up all discovered artifacts:

```text
tts*.wav
subtitles*.srt
```

## Output

The final file is named after the original video filename from `job.yaml -> source.filename`:

```text
<job_dir>/<job-id>/<name>.mp4
```

`<name>` is the original filename without the extension. For example, for `lecture.mov`, the result is:

```text
lecture.mp4
```

After a successful build, the step creates a softlink:

```text
<job_dir>/<job-id>/result.mp4 -> <name>.mp4
```

## Mux behavior

- the video stream is taken from the source video;
- source audio streams are preserved if `ffmpeg` can place them into MP4 without ambiguous conversion;
- each non-excluded `tts*.wav` is added as a separate audio track;
- each non-excluded `subtitles*.srt` is added as a soft subtitle track;
- SRT subtitles are not burned into the video;
- for MP4, the subtitle codec must be compatible with the container;
- each added track is tagged with its language via stream metadata: `tts*.wav` and `subtitles*.srt` tracks take the language from their filename (`tts.<lang>…`, `subtitles.<lang>.srt`), and the preserved source audio takes `languages.source` from `job.yaml`;
- input artifacts are not modified.

Codec choices for MP4:

- the video stream is copied (`-c:v copy`);
- the preserved source audio is copied (`-c:a copy`) when present;
- each `tts*.wav` is encoded to AAC (PCM WAV cannot be placed in MP4);
- subtitles use `mov_text`.

The addition order must be deterministic: discovered `tts*.wav` and `subtitles*.srt` artifacts are sorted by filename.

## Artifact ownership

`mux` owns only:

```text
<name>.mp4
result.mp4
```

It does not modify artifacts from previous steps.

## Idempotency and `--force`

`mux` must be idempotent.

If `<name>.mp4` already exists and `result.mp4` points to it, the step must not recreate the result without `--force`.

If `<name>.mp4` exists, but `result.mp4` is missing or points elsewhere, the step must restore the symlink without rebuilding the video.

`--force` recreates both the final MP4 and `result.mp4`.

## Errors

Expected errors:

- job not found;
- missing `job.yaml`;
- invalid `job.yaml` format;
- missing `source.artifact` in `job.yaml`;
- missing `source.filename` in `job.yaml`;
- missing source video;
- `--exclude` refers to a non-existing artifact;
- `--exclude` refers to something other than `tts*.wav` or `subtitles*.srt`;
- `ffmpeg` is unavailable;
- `ffmpeg` exits with an error;
- unable to write the final MP4;
- unable to create or update `result.mp4`.

Expected errors must be converted into clear CLI errors through the project's base error class.

## Tests

Tests for `mux` must cover:

- reading an existing job;
- reading `source.artifact` and `source.filename` from `job.yaml`;
- finding the source video;
- finding all `tts*.wav` artifacts;
- finding all `subtitles*.srt` artifacts;
- sorting discovered artifacts by filename;
- creating `<name>.mp4` from `source.filename`;
- creating `result.mp4` as a softlink to `<name>.mp4`;
- adding all discovered TTS audio tracks;
- adding all discovered subtitle tracks;
- tagging each track with its language from the artifact filename;
- excluding an audio artifact through `--exclude`;
- excluding a subtitle artifact through `--exclude`;
- error for a non-existing `--exclude`;
- error for an invalid `--exclude`;
- leaving input artifacts unchanged;
- skipping the step if the result already exists;
- restoring a missing or incorrect `result.mp4`;
- recreating the result with `--force`;
- error when the source video is missing;
- error when `ffmpeg` fails.
