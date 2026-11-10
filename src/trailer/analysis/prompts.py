"""Loading of prompt templates stored outside of Python code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from trailer.domain.exceptions import ValidationError

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

KNOWN_PROMPTS = ("analyze_block", "build_content_map", "create_trailer")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the raw markdown prompt template with the given name."""
    if name not in KNOWN_PROMPTS:
        raise ValidationError(f"Неизвестный промпт: {name!r}")
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise ValidationError(f"Файл промпта не найден: {path}")
    return path.read_text(encoding="utf-8")
