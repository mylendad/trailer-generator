from __future__ import annotations

import logging
import re

from trailer.config import BlockSettings
from trailer.domain.models import PreparedTranscript, TranscriptBlock

logger = logging.getLogger(__name__)

_STAGE = "segment"

_SENTENCE_END_RE = re.compile(r"[.!?…]\s*$")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping their trailing punctuation."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class TranscriptSegmenter:
    """Splits a prepared transcript into semantic blocks for LLM analysis.

    Blocks never exceed ``max_block_chars`` or ``max_block_seconds``. Block
    boundaries fall on sentence ends, long pauses and topic-shift markers, so
    sentences and thoughts are not torn apart without need.
    """

    def __init__(self, settings: BlockSettings) -> None:
        self._settings = settings

    def segment(self, prepared: PreparedTranscript) -> list[TranscriptBlock]:
        blocks: list[TranscriptBlock] = []
        current_start: float | None = None
        current_end = 0.0
        current_sentences: list[str] = []
        current_text_start = 0.0

        def flush() -> None:
            nonlocal current_start, current_sentences
            if current_start is None or not current_sentences:
                return
            blocks.append(
                TranscriptBlock(
                    block_id=f"block_{len(blocks) + 1:03d}",
                    start=current_start,
                    end=current_end,
                    text=" ".join(current_sentences),
                )
            )
            current_start = None
            current_sentences = []

        for entry in prepared.entries:
            sentences = _split_sentences(entry.text)
            if not sentences:
                continue
            if current_start is None:
                current_start = entry.start
                current_text_start = entry.start
            gap = entry.start - current_end
            if gap > self._settings.max_pause_inside_block:
                flush()
                current_start = entry.start
                current_text_start = entry.start

            for sentence in sentences:
                projected_len = len(" ".join(current_sentences + [sentence]))
                projected_end = entry.end
                duration = projected_end - current_text_start
                if (current_sentences and projected_len > self._settings.max_block_chars) or (
                    current_sentences and duration > self._settings.max_block_seconds
                ):
                    flush()
                    current_start = entry.start
                    current_text_start = entry.start
                current_sentences.append(sentence)
                current_end = entry.end
        flush()

        if not blocks:
            raise ValueError("Не удалось разбить транскрипцию на блоки")
        total_seconds = blocks[-1].end - blocks[0].start
        logger.info(
            "Разбиение на блоки: %d блоков (средняя длительность %.0f с, всего %.0f с)",
            len(blocks),
            total_seconds / len(blocks),
            total_seconds,
        )
        return blocks
