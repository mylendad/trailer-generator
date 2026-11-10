"""Transcript providers: turning a video into timed text."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trailer.domain.models import Transcript


class TranscriptProvider(ABC):
    """A provider of transcripts (YouTube subtitles now, Whisper later)."""

    @abstractmethod
    def fetch(self, video_id: str, language: str) -> Transcript:
        """Return the raw transcript of the video in the requested language."""
