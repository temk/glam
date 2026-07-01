from pathlib import Path

from glam.errors import GlamError


class GlossaryError(GlamError):
    pass


def load_glossary(path):
    """Load a plain list of never-translate terms — one per line, '#' comments and blank lines ignored."""
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        raise GlossaryError(f"glossary file not found: {path}")

    terms = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            terms.append(line)
    return terms
