# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Project overview

**GLAM** (Glossary-Locked Audio Muxer) is a CLI pipeline for translating videos, mostly English technical and educational content, into Russian or another target language.

The core purpose of the project is glossary-locked translation: important technical terms must either remain untranslated or be translated strictly according to the glossary rules.

The project architecture is described in `docs/architecture.md`.
This file should contain only practical rules, commands, and important constraints.

## Required rules

- First inspect the existing code and follow the current project patterns.
- Always read `docs/architecture.md` before starting work — it extends this file.
- `docs/architecture.md` is the source of truth. When the code and the documentation diverge, the code is wrong: fix the code to match the documentation. Do not edit the documentation to match diverging code without an explicit command.
- Before working on a pipeline step `<step>`, always read the corresponding doc `docs/steps/<step>.md`.
- You may read any repository file when it helps complete the task.
- Do not read or print secrets from `.env*`, `conf/*.local.*`, or other local secret/config files unless explicitly necessary.
- Do not put real secrets into committed config files.
- Do not ask for permission to make normal file edits when the task already requires changes.
- If there are several roughly equivalent architectural or behavioral solutions, briefly describe the options and ask which one to choose.
- For small implementation choices, decide yourself and mention the decision in the final summary.
- When an instruction is scoped (for example "replace X instead of Y", "change only Z"), edit only the named target and leave neighboring code and working branches untouched. If a broadly worded request seems to conflict with that scope, ask one clarifying question instead of silently deleting working code.
- After completing a task, show which files changed and give a brief summary of the changes.
- Do not commit files to git without an explicit command.
- Never push changes to git.

## Python style

- Keep imports at the top of the file.
- Keep package `__init__.py` files empty: no code, no imports, no re-exports. Import each symbol from the submodule that defines it (for example `from glam.common.errors import GlamError`).
- Use Ruff for formatting, linting, and import sorting.
- The Ruff line length is 120 (`line-length` in `pyproject.toml`). Keep a call, exception, or string on a single line when it fits within 120 columns; let `ruff format` decide the wrapping rather than hand-wrapping short constructor calls. If a line still overflows, prefer shortening the expression or message over splitting a simple call across several lines.
- Imports form two groups separated by a blank line: external imports (stdlib and third-party together) first, then local imports from `glam`.
- Within each group, `import x` statements go first, `from x import y` statements after them.
- Sorting imports by line length is preferred but not required (`ruff check --fix` applies it automatically).
- Run `uv run ruff format .` after editing Python files.
- Run `uv run ruff check . --fix` after editing Python files.
- Keep functions small and focused.
- Choose the implementation with the least reasonable nesting; prefer guard clauses and early returns while preserving behavior.
- Use type hints for public functions and non-trivial helpers.
- Add a docstring to a public function or class only when its name and signature do not already make its purpose clear; a self-explanatory name needs no docstring, and a docstring that only restates the name must be removed.
- Private helpers need docstrings only when the intent or behavior is not obvious.
- Do not require docstrings for every small private function.
- Do not add comments that merely restate the code.
- Prefer readable names over explanatory comments.
- Add comments for non-obvious decisions, edge cases, protocol quirks, or workarounds.
- Do not add comments/class docstring in dataclasses it unless strictly necessary.
- Do not mix formatting-only changes with behavioral changes unless asked.

## Serialization

- Use `dacite` for deserializing plain mappings (parsed YAML/JSON) into dataclasses; do not hand-write `from_dict` parsers for new dataclasses.
- Use `dataclasses.asdict` for the reverse direction when serializing dataclasses back to mappings.
- Convert `dacite` errors into the project's base error class from `common.py`, not raw `DaciteError`.

## Commands

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run mypy
uv run pytest -q
```

If you change CLI flags, first check the command signatures in `src/glam/cli.py` instead of assuming that all commands accept the same options.

## Project structure

- `src/glam/` — source code.
- `src/glam/steps/` — pipeline step implementations, one step per file.
- `src/glam/backend/` — per-step model-backend packages (`tts/`, `transcribe/`, `translate/`); each has a `base.py` interface + factory and one module per `protocol` (`openai.py`, `chatterbox.py`, ...). Heavy SDK imports live here, not in `common/`.
- `src/glam/media.py` — wrappers around `ffmpeg`/`ffprobe` via `subprocess`.
- `src/glam/common/` — shared package used by multiple steps; must stay free of heavy imports like the OpenAI SDK.
  - `src/glam/common/errors.py` — the base error class `GlamError`.
  - `src/glam/common/config.py` — the pipeline config file dataclasses (`Config`, `ServiceConfig`) and `read_config`. `ServiceConfig` has required `name`/`protocol`/`url` plus a `params` mapping (default `{}`); the loader keeps `params` opaque and each backend deserializes it into its own typed config.
  - `src/glam/common/job.py` — the job manifest (`job.yaml`) dataclasses and its read/write helpers.
  - `src/glam/common/translation.py` — reading/validating the `translation.<target>.json` artifact shared by `subtitles` and `tts`.
- `tests/` — unit and smoke-style tests.

## Tests

Run relevant tests after changes.

Minimum full test run:

```bash
uv run pytest -q
```

If tests were not run, explicitly say so in the final response.
