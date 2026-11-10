"""Video sources: fetching metadata and downloading the file."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from trailer.domain.models import VideoInfo


class VideoSource(ABC):
    """A provider of source videos (YouTube now, VK etc. later)."""

    @abstractmethod
    def fetch_info(self, url: str) -> VideoInfo:
        """Return metadata of the video behind ``url``."""

    @abstractmethod
    def download(self, url: str, destination: Path) -> Path:
        """Download the video into ``destination`` and return the file path."""
