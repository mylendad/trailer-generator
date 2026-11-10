from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import yt_dlp
from yt_dlp.utils import DownloadError

from trailer.config import DownloadSettings
from trailer.domain.exceptions import InvalidURLError, SourceError
from trailer.domain.models import VideoInfo
from trailer.editing.ffmpeg_binary import resolve_ffmpeg
from trailer.sources.base import VideoSource

logger = logging.getLogger(__name__)

_STAGE = "download"

_NO_FFMPEG_FORMAT = "b[height<=1080][ext=mp4]/b"


def _find_downloaded_file(stem: Path) -> Path | None:
    """Pick the newest real yt-dlp output whose extension it chose itself."""
    candidates = [
        path
        for path in stem.parent.glob(f"{stem.name}.*")
        if path.is_file()
        and ".part" not in path.name
        and ".ytdl" not in path.name
        and ".temp." not in path.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


class YouTubeSource(VideoSource):
    """YouTube video source built on top of ``yt-dlp``."""

    def __init__(self, settings: DownloadSettings) -> None:
        self._settings = settings

    def fetch_info(self, url: str) -> VideoInfo:
        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(cast(Any, options)) as ydl:
                info = ydl.extract_info(url, download=False)
        except DownloadError as exc:
            raise SourceError(_STAGE, f"Не удалось получить информацию о видео: {exc}") from exc
        if info is None or not info.get("id"):
            raise InvalidURLError(url)
        if info.get("_type") == "playlist":
            entries = [entry for entry in info.get("entries") or []]
            info = entries[0] if entries else None
            if info is None:
                raise InvalidURLError(url)
        return VideoInfo(
            video_id=str(info["id"]),
            title=str(info.get("title") or info["id"]),
            duration=float(info.get("duration") or 0.0),
            url=str(info.get("webpage_url") or url),
        )

    def download(self, url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = int(self._settings.max_filesize_gb * 1024**3)
        ffmpeg = resolve_ffmpeg()
        format_selector = self._settings.format
        if ffmpeg is None:
            format_selector = _NO_FFMPEG_FORMAT
        stem = destination.with_suffix("")
        options: dict[str, Any] = {
            "format": format_selector,
            # Let yt-dlp choose the container extension: merged streams may not
            # fit into the extension implied by the destination name.
            "outtmpl": f"{stem}.%(ext)s",
            "max_filesize": max_bytes,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
        }
        if ffmpeg is not None:
            options["ffmpeg_location"] = ffmpeg
        try:
            with yt_dlp.YoutubeDL(cast(Any, options)) as ydl:
                ydl.download([url])
        except DownloadError as exc:
            raise SourceError(_STAGE, f"Не удалось загрузить видео: {exc}") from exc
        produced = _find_downloaded_file(stem)
        if produced is None:
            raise SourceError(_STAGE, f"Загруженный файл отсутствует или пуст: {destination}")
        if produced != destination:
            produced.replace(destination)
        if destination.stat().st_size == 0:
            raise SourceError(_STAGE, f"Загруженный файл пуст: {destination}")
        logger.info("Видео сохранено: %s (%.1f МБ)", destination, destination.stat().st_size / 1e6)
        return destination
