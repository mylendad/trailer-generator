from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_ffmpeg() -> str | None:
    """Return a usable FFmpeg binary path.

    Prefers a system-wide ``ffmpeg`` from PATH and falls back to the binary
    bundled with ``imageio-ffmpeg`` so the pipeline works without installing
    FFmpeg manually.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.warning("Встроенный FFmpeg (imageio-ffmpeg) недоступен: %s", exc)
        return None
    if not Path(bundled).is_file():
        return None
    logger.debug("Используется встроенный FFmpeg: %s", bundled)
    return bundled
