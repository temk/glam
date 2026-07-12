import json
import itertools
from dacite import DaciteError
from dacite import from_dict as dacite_from_dict
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import asdict, dataclass

from glam.common.job import JOB_MANIFEST_NAME, read_job_manifest
from glam.common.hooks import service_hooks
from glam.common.config import Config, ServiceName, ServiceConfig
from glam.common.errors import GlamError
from glam.common.translation import translation_filename
from glam.backend.translate.base import ChatResult, TranslateBackend, TranslateBackendError, build_translate_backend

TRANSCRIPT_NAME = "transcript.json"
GLOSSARY_NAME = "glossary.json"

# `--dump` writes one file per request/response into this per-language folder (`{}` is the target
# language), numbered in request order. The file is written before the response is parsed, so a parse
# failure (e.g. unknown ids) still leaves it on disk. A debug artifact; no step reads it.
DUMP_DIR_TEMPLATE = "translate.{}.dump"
DUMP_FILE_TEMPLATE = "{:05d}.json"

# Each translated segment is cached here as its own JSON so a crash (or `--start`) can resume without
# re-translating finished work. Files are keyed by target so several languages coexist in one job.
TRANSLATE_CACHE_DIRNAME = "translate"

# Default segments translated per request. One keeps the id-to-text mapping unambiguous: the model
# cannot re-segment a single-item batch, which is the only fully drift-proof case (see `_translate`).
BATCH_SIZE = 1
# Default number of already-translated preceding segments sent read-only ahead of each batch.
CONTEXT_SIZE = 5
# Default number of following source segments given as read-only source context, so the model can see
# how each sentence ends before translating its opening (word order differs across languages).
LOOKAHEAD_SIZE = 2

# Constrained decoding: providers that honor it (Ollama, vLLM, OpenAI) can only emit tokens
# matching this schema, which rules out the malformed JSON weak models otherwise produce.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "translations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "translated_text": {"type": "string"}},
                        "required": ["id", "translated_text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["segments"],
            "additionalProperties": False,
        },
    },
}


class TranslateError(GlamError):
    pass


@dataclass
class TranslatedSegment:
    id: int
    start: float
    end: float
    text: str
    translated_text: str | None = None


@dataclass
class Translation:
    """A transcript augmented in place with `translated_text` on every segment.

    Top-level fields are carried over from `transcript.json` untouched, so dropping
    every `translated_text` yields a document identical to the source transcript.
    """

    version: int
    step: str
    job_id: str
    source_language: str
    model: str
    audio_artifact: str
    segments: list[TranslatedSegment]


def run(
    job_id: str,
    config: Config,
    target: str | None = None,
    force: bool = False,
    echo=print,
    batch_size: int = BATCH_SIZE,
    context_size: int = CONTEXT_SIZE,
    lookahead_size: int = LOOKAHEAD_SIZE,
    start: int | None = None,
    dump: bool = False,
) -> Path:
    """Translate a job's transcript through the configured LLM service into `translation.<target>.json`."""
    job_path = config.job_dir / job_id
    if not job_path.is_dir():
        raise TranslateError(f"job not found: {job_id} (looked in {job_path})")

    manifest = read_job_manifest(job_path / JOB_MANIFEST_NAME)
    target = target or manifest.languages.target
    if not target:
        raise TranslateError("missing target language: pass --target or set languages.target in job.yaml")
    translation = _load_transcript(job_path / TRANSCRIPT_NAME)
    glossary = _load_glossary(job_path / GLOSSARY_NAME)

    translation_path = job_path / translation_filename(target)
    # `--start` (like `--force`) means "run anyway": don't skip on an existing output.
    if translation_path.exists() and not force and start is None:
        echo(f"skip translation, already exists: {translation_path}")
        return translation_path

    service = config[ServiceName.TRANSLATE]
    dump_dir = job_path / DUMP_DIR_TEMPLATE.format(target) if dump else None
    with service_hooks(service.hooks, echo):
        _apply_translations(
            translation,
            manifest.languages.source,
            target,
            glossary,
            service,
            echo,
            batch_size,
            context_size,
            lookahead_size,
            job_path / TRANSLATE_CACHE_DIRNAME,
            target,
            force,
            start,
            dump_dir,
        )
        translation_path.write_text(json.dumps(asdict(translation), ensure_ascii=False, indent=2) + "\n")

    echo(f"wrote {translation_path} ({len(translation.segments)} segments)")
    return translation_path


def _load_transcript(path: Path) -> Translation:
    if not path.exists():
        raise TranslateError(f"missing transcript artifact: {path}")
    try:
        data = json.loads(path.read_text())
        return dacite_from_dict(data_class=Translation, data=data)
    except (json.JSONDecodeError, DaciteError) as e:
        raise TranslateError(f"invalid transcript {path}: {e}") from e


def _load_glossary(path: Path) -> dict[str, str]:
    if not path.exists():
        raise TranslateError(f"missing glossary artifact: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise TranslateError(f"invalid glossary {path}: {e}") from e
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise TranslateError(f"invalid glossary {path}: must be a JSON object mapping strings to strings")
    return data


def _apply_translations(
    translation: Translation,
    source: str,
    target: str,
    glossary: dict[str, str],
    service: ServiceConfig,
    echo,
    batch_size: int,
    context_size: int,
    lookahead_size: int,
    cache_dir: Path,
    cache_key: str,
    force: bool,
    start: int | None,
    dump_dir: Path | None,
) -> None:
    system_prompt = _system_prompt(source, target, glossary)
    segments = translation.segments
    total = len(segments)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fill in segments already on disk; `--start` marks earlier ones as done (they must be cached).
    _apply_cache(segments, cache_dir, cache_key, force, start)
    pending = [(index, segment) for index, segment in enumerate(segments) if segment.translated_text is None]
    if not pending:
        echo("all segments already translated (from cache)")
        return

    # One file per request, numbered in request order across the whole run.
    if dump_dir is not None:
        _prepare_dump_dir(dump_dir)
    dump_counter = itertools.count(1)

    backend = build_translate_backend(service)  # built only when there is work to do
    for group_start in range(0, len(pending), batch_size):
        chunk = pending[group_start : group_start + batch_size]
        batch = [segment for _, segment in chunk]
        first, last = chunk[0][0], chunk[-1][0]
        # The segments immediately before and after the batch, in the source language, given as separate
        # reference so the model sees where each sentence begins and ends without confusing that
        # surrounding text with the batch it must translate. `before` doubles as the target-language
        # continuity context (its already-translated text).
        before = segments[max(0, first - context_size) : first]
        after = segments[last + 1 : min(total, last + 1 + lookahead_size)]
        echo(f"[{datetime.now():%H:%M:%S}] translating segments {first + 1}-{last + 1} of {total}")
        translations = _translate(backend, batch, before, after, system_prompt, dump_dir, dump_counter)
        for segment in batch:
            segment.translated_text = translations[segment.id]
        _cache_segments(batch, cache_dir, cache_key)


def _prepare_dump_dir(path: Path) -> None:
    """Create the dump folder and drop stale files so it reflects only the current run."""
    path.mkdir(parents=True, exist_ok=True)
    for stale in path.glob("*.json"):
        stale.unlink()


def _cache_path(cache_dir: Path, cache_key: str, segment_id: int) -> Path:
    return cache_dir / f"{cache_key}.{segment_id:05d}.json"


def _apply_cache(
    segments: list[TranslatedSegment], cache_dir: Path, cache_key: str, force: bool, start: int | None
) -> None:
    for index, segment in enumerate(segments):
        position = index + 1
        cache_path = _cache_path(cache_dir, cache_key, segment.id)
        if start is not None and position < start:
            if not cache_path.exists():
                raise TranslateError(
                    f"--start {start}: segment {position} (id {segment.id}) is not cached at {cache_path}; "
                    "use a lower --start or run without it"
                )
            segment.translated_text = _read_cache(cache_path)
        elif cache_path.exists() and not force:
            segment.translated_text = _read_cache(cache_path)


def _read_cache(path: Path) -> str:
    try:
        return str(json.loads(path.read_text())["translated_text"])
    except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
        raise TranslateError(f"invalid translation cache {path}: {e}") from e


def _cache_segments(segments: list[TranslatedSegment], cache_dir: Path, cache_key: str) -> None:
    for segment in segments:
        path = _cache_path(cache_dir, cache_key, segment.id)
        record = {"id": segment.id, "translated_text": segment.translated_text}
        try:
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            raise TranslateError(f"unable to cache segment {segment.id} to {path}: {e}") from e


def _system_prompt(source: str, target: str, glossary: dict[str, str]) -> str:
    lines = [
        f"You translate subtitle segments from {source} to {target} for technical and educational videos.",
        "The user message is a JSON object with these fields:",
        "- 'translate': the segments you must translate, as an array of {id, text}. This is the ONLY text to "
        "translate.",
        "- 'text_before': the source-language text right BEFORE these segments — reference for context only; do NOT "
        "translate or return it;",
        "- 'text_after': the source-language text right AFTER these segments — reference for context only; do NOT "
        "translate or return it;",
        "- 'translated_before': the target-language translation of 'text_before' (how the preceding text was "
        "already translated), so you keep terminology and wording consistent — reference only; do NOT translate or "
        "return it.",
        'Reply with ONLY a JSON object of the form {"segments": [{"id": <int>, "translated_text": "<translation>"}]}.',
        "Return exactly one entry per 'translate' item, in the same order, reusing its id. Translate ONLY that "
        "item's own text — never fold in words from 'text_before' or 'text_after'. Never merge, split, add, drop, "
        "or reorder segments.",
    ]
    if glossary:
        lines.append("Apply this glossary strictly wherever a term appears; render each term exactly as specified:")
        lines.extend(f"- {term!r} -> {rendering!r}" for term, rendering in glossary.items())
    else:
        lines.append("No glossary is provided; translate normally.")
    return "\n".join(lines)


def _translate(
    backend: TranslateBackend,
    segments: list[TranslatedSegment],
    before: list[TranslatedSegment],
    after: list[TranslatedSegment],
    system_prompt: str,
    dump_dir: Path | None,
    dump_counter: "itertools.count[int]",
) -> dict[int, str]:
    """Translate a batch, trusting the model's ids only when the reply is an exact 1:1 match.

    Weak models re-segment: they merge or split fragments and renumber the outputs 0,1,2,… so the ids
    stop labelling the source segments (this silently misaligns translations). When the returned ids are
    not exactly the requested ones, the labels are untrustworthy, so the batch is split and each half
    retried — down to a single segment, whose lone translation is taken by position (its returned id is
    ignored), which is the only unambiguous case.
    """
    requested_ids = [segment.id for segment in segments]
    pairs = _request(backend, segments, before, after, system_prompt, dump_dir, dump_counter)
    if Counter(pid for pid, _ in pairs) == Counter(requested_ids):
        return dict(pairs)

    if len(segments) == 1:
        only = segments[0]
        if not pairs:
            raise TranslateError(f"model returned no translation for segment {only.id}")
        return {only.id: pairs[0][1]}

    mid = len(segments) // 2
    left = _translate(backend, segments[:mid], before, after, system_prompt, dump_dir, dump_counter)
    right = _translate(backend, segments[mid:], before, after, system_prompt, dump_dir, dump_counter)
    return {**left, **right}


def _request(
    backend: TranslateBackend,
    segments: list[TranslatedSegment],
    before: list[TranslatedSegment],
    after: list[TranslatedSegment],
    system_prompt: str,
    dump_dir: Path | None,
    dump_counter: "itertools.count[int]",
) -> list[tuple[int, str]]:
    payload = {
        # The only thing to translate and return, keyed by id.
        "translate": [{"id": segment.id, "text": segment.text} for segment in segments],
        # Surrounding source-language text, kept separate from the batch so the model can read whole
        # sentences (start and end) without mistaking this context for text it must translate.
        "text_before": " ".join(segment.text.strip() for segment in before),
        "text_after": " ".join(segment.text.strip() for segment in after),
        # The target-language translation of `text_before`, for terminology and wording continuity.
        "translated_before": " ".join(segment.translated_text.strip() for segment in before if segment.translated_text),
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        result = backend.complete(messages, RESPONSE_FORMAT)
    except TranslateBackendError as e:
        _dump(dump_dir, dump_counter, segments, messages, error=str(e))
        raise
    # Dump before parsing, so an id mismatch that triggers a split still leaves the exchange on disk.
    _dump(dump_dir, dump_counter, segments, messages, content=result.content, finish_reason=result.finish_reason)
    return _parse_response(result)


def _returned_ids(content: str | None) -> list[int] | None:
    """Best-effort ids from a raw response, so a dump shows the mismatch even when parsing later fails."""
    if not content:
        return None
    try:
        return [int(item["id"]) for item in json.loads(content)["segments"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _dump(dump_dir, dump_counter, segments, messages, content=None, finish_reason=None, error=None) -> None:
    """Write this single request/response exchange to its own numbered file when dumping."""
    if dump_dir is None:
        return
    record = {
        "requested_ids": sorted(segment.id for segment in segments),
        "returned_ids": _returned_ids(content),
        "request": {"messages": messages},
        "response": {"content": content, "finish_reason": finish_reason, "error": error},
    }
    path = dump_dir / DUMP_FILE_TEMPLATE.format(next(dump_counter))
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def _parse_response(result: ChatResult) -> list[tuple[int, str]]:
    """Parse the reply into ordered (id, translated_text) pairs. Whether the ids match what was
    requested is judged by `_translate` (a mismatch triggers a split), not here."""
    content = result.content
    if not content:
        raise TranslateError("translation response has no content")
    if result.finish_reason == "length":
        raise TranslateError("translation response was cut off by the output token limit; try a smaller --batch-size")
    try:
        items = json.loads(content)["segments"]
        return [(int(item["id"]), str(item["translated_text"])) for item in items]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise TranslateError(f"invalid translation response: {e}; content starts with {content[:120]!r}") from e
