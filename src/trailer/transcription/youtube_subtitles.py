from __future__ import annotations

import logging

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from trailer.domain.exceptions import TranscriptionError
from trailer.domain.models import Transcript, TranscriptEntry
from trailer.transcription.base import TranscriptProvider

logger = logging.getLogger(__name__)

_STAGE = "transcript"


class YouTubeSubtitlesProvider(TranscriptProvider):
    """Fetches YouTube subtitles, preferring manual over auto-generated ones."""

    def __init__(self) -> None:
        self._api = YouTubeTranscriptApi()

    def fetch(self, video_id: str, language: str) -> Transcript:
        try:
            transcript_list = self._api.list(video_id)
        except TranscriptsDisabled as exc:
            raise TranscriptionError(_STAGE, "Субтитры для этого видео отключены") from exc
        except Exception as exc:  # youtube_transcript_api raises various errors
            raise TranscriptionError(
                _STAGE, f"Не удалось получить список субтитров: {exc}"
            ) from exc

        try:
            # Prefer manually created subtitles, fall back to auto-generated.
            try:
                fetched = transcript_list.find_transcript([language]).fetch()
            except NoTranscriptFound:
                fetched = transcript_list.find_generated_transcript([language]).fetch()
        except NoTranscriptFound as exc:
            available = ", ".join(t.language_code for t in transcript_list)
            raise TranscriptionError(
                _STAGE,
                f"Субтитры на языке '{language}' не найдены. Доступные языки: {available or 'нет'}",
            ) from exc
        except TranscriptsDisabled as exc:
            raise TranscriptionError(_STAGE, "Субтитры для этого видео отключены") from exc
        except Exception as exc:
            raise TranscriptionError(_STAGE, f"Не удалось получить субтитры: {exc}") from exc

        entries = [
            TranscriptEntry(
                start=snippet.start, duration=max(snippet.duration, 0.01), text=snippet.text
            )
            for snippet in fetched
            if snippet.text and snippet.text.strip()
        ]
        if not entries:
            raise TranscriptionError(_STAGE, "Полученная транскрипция пуста")
        logger.info(
            "Получена транскрипция: %d реплик, %.0f секунд",
            len(entries),
            entries[-1].end - entries[0].start,
        )
        return Transcript(language=language, entries=entries)
