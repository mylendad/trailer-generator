"""Storage of intro/outro templates managed via the Telegram bot."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from trailer.domain.exceptions import EditingError

logger = logging.getLogger(__name__)

_NAME_SAFE = re.compile(r"[^\w.\-]+", re.UNICODE)

_VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")


class MediaTemplates:
    """Keeps uploaded intro/outro videos and activates one for rendering.

    Templates live in ``templates_dir``; the chosen template is copied into
    ``active_dir`` (which the render stage scans for the first ``*.mp4``).
    """

    def __init__(self, templates_dir: Path, active_dir: Path) -> None:
        self._templates_dir = templates_dir
        self._active_dir = active_dir

    def list(self) -> list[str]:
        if not self._templates_dir.is_dir():
            return []
        return sorted(path.stem for path in self._templates_dir.glob("*.mp4"))

    def active(self) -> Path | None:
        if not self._active_dir.is_dir():
            return None
        files = sorted(self._active_dir.glob("*.mp4"))
        return files[0] if files else None

    def save(self, name: str, source: Path) -> Path:
        if source.suffix.lower() not in _VIDEO_SUFFIXES:
            raise EditingError("bot", f"Неподдерживаемый формат интро: {source.suffix}")
        target = self._sanitize_target(name)
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        destination = self._templates_dir / target
        shutil.copyfile(source, destination)
        logger.info("Шаблон интро сохранён: %s", destination)
        return destination

    def activate(self, name: str) -> Path:
        source = self._templates_dir / self._sanitize_target(name)
        if not source.exists():
            raise EditingError("bot", f"Шаблон не найден: {name}")
        self._active_dir.mkdir(parents=True, exist_ok=True)
        for path in self._active_dir.glob("*.mp4"):
            path.unlink()
        active = self._active_dir / source.name
        shutil.copyfile(source, active)
        logger.info("Активное интро заменено: %s", active)
        return active

    def _sanitize_target(self, name: str) -> str:
        stem = _NAME_SAFE.sub("_", Path(name).stem).strip("._-") or "intro"
        return f"{stem}.mp4"
