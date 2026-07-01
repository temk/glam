import json
import re
from pathlib import Path

import openai

from glam.clients import build_openai_client
from glam.config import step_config
from glam.errors import GlamError
from glam.glossary import load_glossary
from glam.paths import job_dir, resolve_artifact, slugify

LANG_NAMES = {
    "ru": "Russian",
}


class TranslateError(GlamError):
    pass


def run(video_id, config, lang, jobs_root=Path("jobs"), transcript_path=None, force=False, echo=print):
    job_path = job_dir(jobs_root, video_id)
    translate_cfg = step_config(config, "translate")
    model = translate_cfg["model"]

    output_path = job_path / f"translation.{lang}.{slugify(model)}.json"
    if output_path.exists() and not force:
        echo(f"skip translation, already exists: {output_path}")
        return output_path

    transcript_path = _resolve_transcript(job_path, config, transcript_path)
    transcript = json.loads(Path(transcript_path).read_text())
    segments = transcript.get("segments") or []
    if not segments:
        raise TranslateError(f"no segments found in {transcript_path}")

    glossary_terms = load_glossary(translate_cfg.get("glossary"))
    batch_size = translate_cfg.get("batch_size", 20)
    overlap = translate_cfg.get("overlap", 2)
    use_structured_output = translate_cfg.get("structured_output", True)

    client = build_openai_client(translate_cfg)

    translated = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        context = translated[-overlap:] if translated and overlap > 0 else []
        echo(f"translating segments {i + 1}-{i + len(batch)} of {len(segments)}")
        texts = _translate_batch(
            client, model, lang, glossary_terms, context, batch, use_structured_output
        )
        for seg, text in zip(batch, texts):
            translated.append({
                "id": seg.get("id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": text,
                "text_source": (seg.get("text") or "").strip(),
            })

    output = {"model": model, "lang": lang, "segments": translated}
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    echo(f"wrote {output_path}")
    return output_path


def _resolve_transcript(job_path, config, explicit_path):
    asr_cfg = config.get("steps", {}).get("asr")
    exact_candidate = None
    if asr_cfg and asr_cfg.get("model"):
        exact_candidate = job_path / f"transcript.{slugify(asr_cfg['model'])}.json"
    return resolve_artifact(
        job_path, "transcript.*.json", explicit_path,
        exact_candidate=exact_candidate, hint="run 'glam transcribe' first",
    )


def _translate_batch(client, model, lang, glossary_terms, context, batch, use_structured_output):
    messages = [
        {"role": "system", "content": _system_prompt(lang, glossary_terms)},
        {"role": "user", "content": _user_prompt(context, batch)},
    ]
    kwargs = {"model": model, "messages": messages}
    if use_structured_output:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except openai.OpenAIError as e:
        raise TranslateError(
            f"translation request failed: {e}. Verify the backend is reachable and speaks "
            "the OpenAI-compatible /v1/chat/completions shape before assuming this is a code bug."
        ) from e

    content = response.choices[0].message.content
    return _parse_translations(content, len(batch))


def _parse_translations(content, expected_count):
    data = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    translations = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(translations, list) or len(translations) != expected_count:
        got = len(translations) if isinstance(translations, list) else "invalid"
        raise TranslateError(
            f"expected {expected_count} translations, got {got} — model response: {content[:500]!r}"
        )
    return translations


def _system_prompt(lang, glossary_terms):
    lang_name = LANG_NAMES.get(lang, lang)
    lines = [
        f"You are a professional technical translator. Translate English video-transcript "
        f"segments into {lang_name}.",
        "You will receive numbered segments and must return exactly that many translations, "
        "in the same order — never merge, split, drop, or reorder segments.",
        "Preserve established English technical loanwords that Russian-speaking practitioners "
        "commonly use in English rather than translating (e.g. machine-learning and "
        "software-engineering jargon) — use judgment beyond the explicit list below.",
    ]
    if glossary_terms:
        lines.append(
            "Always leave these exact terms untranslated, in English: "
            + ", ".join(glossary_terms) + "."
        )
    lines.append(
        'Respond with a single JSON object of the form {"translations": ["...", ...]} '
        "and nothing else — no markdown, no commentary."
    )
    return "\n".join(lines)


def _user_prompt(context, batch):
    parts = []
    if context:
        parts.append(
            "Context from the previous batch — already translated, for continuity only, "
            "do not re-translate or include in your output:"
        )
        for j, seg in enumerate(context, start=1):
            parts.append(f"{j}. EN: {seg['text_source']}\n   Translation: {seg['text']}")
        parts.append("")
    parts.append("Segments to translate now:")
    for j, seg in enumerate(batch, start=1):
        parts.append(f"{j}. {(seg.get('text') or '').strip()}")
    return "\n".join(parts)
