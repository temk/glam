import json
from dacite import DaciteError
from dacite import from_dict as dacite_from_dict
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass

from glam.common.job import JOB_MANIFEST_NAME, read_job_manifest
from glam.common.config import Config, ServiceName
from glam.common.errors import GlamError
from glam.common.translation import translation_filename
from glam.backend.translate.base import ChatResult, TranslateBackend, TranslateBackendError, build_translate_backend

TRANSCRIPT_NAME = "transcript.json"
GLOSSARY_NAME = "glossary.json"
# Debug dumps go into a per-language folder (`{}` is the target language, e.g. `translate.ru.dump/`),
# one file per batch inside it, named by the 1-based batch number zero-padded so files sort naturally.
DUMP_DIR_TEMPLATE = "translate.{}.dump"
DUMP_FILE_TEMPLATE = "{:03d}.json"

# Default segments translated per request.
BATCH_SIZE = 100
# Default number of already-translated preceding segments sent read-only ahead of each batch.
CONTEXT_SIZE = 100

# Weak local models occasionally drop a few segments from a large batch; re-request the
# stragglers this many rounds before giving up.
MAX_ROUNDS = 6

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
    if translation_path.exists() and not force:
        echo(f"skip translation, already exists: {translation_path}")
        return translation_path

    backend = build_translate_backend(config[ServiceName.TRANSLATE])
    # When dumping, each batch writes its own file inside translate.<target>.dump/, rewritten after
    # every request (retries included) so the exchange survives a mid-step crash.
    dump_dir = job_path / DUMP_DIR_TEMPLATE.format(target) if dump else None
    if dump_dir is not None:
        _prepare_dump_dir(dump_dir)
    _apply_translations(
        translation,
        manifest.languages.source,
        target,
        glossary,
        backend,
        echo,
        batch_size,
        context_size,
        dump_dir,
    )

    translation_path.write_text(json.dumps(asdict(translation), ensure_ascii=False, indent=2) + "\n")
    echo(f"wrote {translation_path} ({len(translation.segments)} segments)")
    return translation_path


def _prepare_dump_dir(path: Path) -> None:
    """Create the dump folder and drop stale batch files so it reflects only the current run."""
    path.mkdir(parents=True, exist_ok=True)
    for stale in path.glob("*.json"):
        stale.unlink()


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
    backend: TranslateBackend,
    echo,
    batch_size: int,
    context_size: int,
    dump_dir: Path | None,
) -> None:
    system_prompt = _system_prompt(source, target, glossary)
    segments = translation.segments
    total = len(segments)
    for index, start in enumerate(range(0, total, batch_size), start=1):
        batch = segments[start : start + batch_size]
        # Preceding segments, already translated in earlier batches, give the model
        # target-language continuity. They are read-only and not expected back.
        context = segments[max(0, start - context_size) : start]
        echo(f"[{datetime.now():%H:%M:%S}] translating segments {start + 1}-{start + len(batch)} of {total}")
        dump_path = dump_dir / DUMP_FILE_TEMPLATE.format(index) if dump_dir is not None else None
        translations = _translate(backend, batch, context, system_prompt, dump_path)
        for segment in batch:
            segment.translated_text = translations[segment.id]


def _system_prompt(source: str, target: str, glossary: dict[str, str]) -> str:
    lines = [
        f"You translate subtitle segments from {source} to {target} for technical and educational videos.",
        "The user message is a JSON object with:",
        "- 'context': the already-translated target-language text immediately preceding these segments, as plain "
        "text for continuity and consistent terminology only — do NOT translate or return it;",
        "- 'translate': the segments to translate, as an array of {id, text}.",
        'Reply with ONLY a JSON object of the form {"segments": [{"id": <int>, "translated_text": "<translation>"}]}.',
        "Translate EVERY 'translate' item: return exactly one entry per id, the same number of entries as the input, "
        "reusing the same ids. Never skip, merge, split, add, drop, or reorder segments.",
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
    context: list[TranslatedSegment],
    system_prompt: str,
    dump_path: Path | None,
) -> dict[int, str]:
    by_id = {segment.id: segment for segment in segments}
    result: dict[int, str] = {}
    records: list[dict] = []  # every request/response of this batch, dumped incrementally
    for _ in range(MAX_ROUNDS):
        pending = [by_id[i] for i in by_id if i not in result]
        if not pending:
            return result
        result.update(_request(backend, pending, context, system_prompt, records, dump_path))
    missing = sorted(i for i in by_id if i not in result)
    raise TranslateError(
        f"model did not translate {len(missing)} segment(s) after {MAX_ROUNDS} attempts "
        f"(e.g. ids {missing[:5]}); try a smaller --batch-size"
    )


def _request(
    backend: TranslateBackend,
    segments: list[TranslatedSegment],
    context: list[TranslatedSegment],
    system_prompt: str,
    records: list[dict],
    dump_path: Path | None,
) -> dict[int, str]:
    payload = {
        # Context is bare translated text (not structured), giving the model target-language
        # continuity without inviting it to echo ids back.
        "context": " ".join(segment.translated_text.strip() for segment in context if segment.translated_text),
        "translate": [{"id": segment.id, "text": segment.text} for segment in segments],
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        result = backend.complete(messages, RESPONSE_FORMAT)
    except TranslateBackendError as e:
        _record(records, dump_path, backend.model, segments, messages, error=str(e))
        raise
    _record(
        records,
        dump_path,
        backend.model,
        segments,
        messages,
        content=result.content,
        finish_reason=result.finish_reason,
    )
    return _parse_response(result, {segment.id for segment in segments})


def _record(records, dump_path, model, segments, messages, content=None, finish_reason=None, error=None) -> None:
    """Append this exchange to the batch's records and, when dumping, rewrite its file."""
    records.append(
        {
            "model": model,
            "requested_ids": sorted(segment.id for segment in segments),
            "request": {"messages": messages},
            "response": {"content": content, "finish_reason": finish_reason, "error": error},
        }
    )
    if dump_path is not None:
        dump_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def _parse_response(result: ChatResult, requested_ids: set[int]) -> dict[int, str]:
    content = result.content
    if not content:
        raise TranslateError("translation response has no content")
    if result.finish_reason == "length":
        raise TranslateError("translation response was cut off by the output token limit; try a smaller --batch-size")
    try:
        items = json.loads(content)["segments"]
        translations = {int(item["id"]): str(item["translated_text"]) for item in items}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise TranslateError(f"invalid translation response: {e}; content starts with {content[:120]!r}") from e
    if len(translations) != len(items):
        raise TranslateError("translation response contains duplicate segment ids")
    unknown = sorted(set(translations) - requested_ids)
    if unknown:
        raise TranslateError(f"translation response contains unknown segment ids: {unknown}")
    # Missing ids are not an error here: `_translate` re-requests the stragglers.
    return translations
