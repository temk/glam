# CLAUDE.md

Guidance for Claude Code working in this repository. Full design and rationale: `docs/architecture.md` — read it before making architectural changes, don't duplicate it here.

## What this is

**GLAM** (Glossary-Locked Audio Muxer) — a CLI pipeline that translates video (mainly English technical/educational content) into Russian or another target language: ASR transcription → LLM translation (with a technical-glossary system) → subtitle generation → TTS dubbing → mux back into the video.

## Project state

All six pipeline steps plus `run` are implemented (`src/glam/steps/`) — the full `init → transcribe → translate → subtitles → tts → mux` chain in `docs/architecture.md` is built and passes an end-to-end smoke test (real `ffmpeg`, a stubbed OpenAI-compatible HTTP server standing in for ASR/LLM/TTS). Still worth treating as first-pass: real backends haven't been exercised, and several sub-behaviors (below) were judgment calls made where the architecture doc didn't fully specify one.

- Media I/O goes through the `ffmpeg-python` package (`src/glam/media.py`), not by shelling out to the `ffmpeg`/`ffprobe` binaries directly.
- ASR/LLM/TTS calls go through the `openai` SDK pointed at a per-step `base_url` (`src/glam/clients.py`) — this *is* "the OpenAI-compatible client" referenced in Rules below, not a bespoke HTTP wrapper.
- CLI is built with `click` (`src/glam/cli.py`), one subcommand per pipeline step. All step commands are wrapped in `handle_glam_errors`, which turns any `GlamError` subclass (`src/glam/errors.py`) into a clean one-line CLI error instead of a traceback — reuse that base class for new step errors rather than raising a bare `RuntimeError`.
- All config lives under `conf/`: `conf/config.example.yaml` (portable template, committed) and `conf/config.local.yaml` (real deployment values, gitignored via the `*.local.*` pattern). `-c`/`--config` defaults to `conf/config.yaml` (still not a real file — same "must pass `-c` explicitly" convention as before, just a new directory). Loaded via `glam.config.load_config`/`step_config`. `steps.<name>.api_key_env` names an env var to read the API key from; if omitted, a placeholder key is sent (fine for local servers that don't check it). `subtitles` and `mux` take no `-c` (matches the CLI signatures in architecture.md) — they don't call a model backend, so their tuning knobs are CLI flags, not config.
- Every step that consumes another step's output (translate←transcript, subtitles/tts←translation, mux←tts_track/subtitles) resolves it via `glam.paths.resolve_artifact`: try an exact filename built from the upstream step's configured model, else fall back to globbing the job dir and erroring on ambiguity. An explicit `--transcript`/`--translation`/`--subtitles`/`--tts-track` flag always overrides.
- `translate` (`src/glam/steps/translate.py`) batches segments (`steps.translate.batch_size`, default 20) with the last `overlap` (default 2) already-translated segments passed back as read-only context for continuity across batches — not token-count-based chunking, since that needs a per-model tokenizer this project doesn't have; revisit if a real long video's segment count needs it. `structured_output: false` in config disables `response_format={"type": "json_object"}` for backends that reject it — no automatic fallback/retry is implemented. Glossary format is a plain word list — one term per line, `#` comments allowed, no YAML — loaded by `src/glam/glossary.py`; example at `conf/glossary.example.txt`, referenced from config via `steps.translate.glossary`.
- `subtitles` (`src/glam/resegment.py`) re-times translated segments for reading speed (`--cps`/`--max-chars-per-line`/`--max-lines`, defaults 18/42/2 per architecture.md). A segment's original `[start, end]` is a floor, not a fixed window: text needing more time extends the cue up to the next segment's start (minus a small gap) so cues never overlap; if that's still not enough, the cue is compressed below ideal reading duration rather than cascading re-timing across the rest of the video.
- `tts` (`src/glam/steps/tts.py`) calls `/v1/audio/speech` per segment, corrects duration via `ffmpeg atempo` clamped to `steps.tts.max_atempo` (default 1.3, i.e. ±30%) — beyond that bound the clip is left uncorrected and assembly just accepts the drift for that segment (rather than shifting every later segment), matching the "accept drift" option flagged in architecture.md's open questions. Per-segment clips live in `tts_segments.<lang>.<model>.<voice>/` and are individually reusable across runs (delete one clip and rerun without `--force` to only redo that one); `--force` regenerates every clip and reassembles.
- `mux` re-encodes video only for `--hardsub` (subtitle-burn requires it); otherwise video is stream-copied. Original audio is stream-copied when kept; the dubbed track is always re-encoded to AAC. No stream language/title metadata is set (kept simple rather than guessing language tags without a config source in this command).
- Nothing in this pipeline has been exercised against a real ASR/LLM/TTS backend — every step is unit-tested with a mocked/stubbed `openai` client (`tests/test_*.py`), and the full `run` chain was manually smoke-tested once against a throwaway local HTTP stub (not committed) plus real `ffmpeg`. Per "Verify before assuming" below, confirm against the actual inference hosts — especially whether they honor `response_format`, `timestamp_granularities`, and TTS `response_format=wav` — before trusting this end-to-end.

## Environment

- Runs on the developer's laptop (Linux Mint, bash). `ffmpeg` handles local audio extraction and muxing.
- ASR/LLM/TTS inference runs elsewhere (a local cluster or a commercial API), called over HTTP via OpenAI-compatible endpoints. Never assume local GPU access for these steps.
- The pipeline does not download video — it takes a local file as input.

## Commands

```
uv sync                  # install deps + create .venv
uv run glam init <video_file> [--id ID] [--force] [--jobs-dir DIR]
uv run glam transcribe <video_id> [-c conf/config.local.yaml] [--force] [--jobs-dir DIR]
uv run glam translate <video_id> --lang ru [-c conf/config.local.yaml] [--transcript PATH] [--force] [--jobs-dir DIR]
uv run glam subtitles <video_id> --lang ru [--translation PATH] [--cps N] [--max-chars-per-line N] [--max-lines N] [--force]
uv run glam tts <video_id> --lang ru [-c conf/config.local.yaml] [--translation PATH] [--force]
uv run glam mux <video_id> --lang ru [--hardsub] [--no-keep-original-audio] [--force]
uv run glam run <video_file> --lang ru [-c conf/config.local.yaml] [--hardsub] [--force]
uv run pytest -q
```

## Rules

- Every step is idempotent: skip recompute if the output file already exists, unless `--force`.
- Never overwrite a cached step's output — filenames encode `<lang>.<model>` (or the relevant model). A different backend/model produces a new file, not a replacement.
- ASR/LLM/TTS calls go through the OpenAI-compatible client. Don't add a bespoke per-provider client unless a provider genuinely can't speak that shape.
- Translation must apply the glossary (terms left untranslated) — this is the project's core motivation, don't regress it.

## Verify before assuming

- The configured backends actually expose OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/audio/transcriptions`, `/v1/audio/speech`) — check with a direct request, don't assume.
- The inference host(s) are network-reachable from wherever glam runs before hardcoding a `base_url` into config.
- The local `ffmpeg` build: `atempo` filter present, multi-track MKV muxing works.
