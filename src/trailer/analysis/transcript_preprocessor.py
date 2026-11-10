from __future__ import annotations

import logging
import re

from trailer.domain.models import PreparedEntry, PreparedTranscript, Transcript

logger = logging.getLogger(__name__)

_STAGE = "prepare"

# Auto-subtitle artifacts: [Music], (applause), ♪♪ etc.
_ARTIFACT_RE = re.compile(
    r"[\[\(]{1,2}\s*(музыка|аплодисменты|смех|приветствия|music|applause|laughter|cheering|"
    r"silence|неразборчиво|inaudible)\s*[\]\)]{1,2}",
    re.IGNORECASE,
)
_NOTES_RE = re.compile(r"♪+|♪[^♪]*♪|>>|->")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = _ARTIFACT_RE.sub(" ", text)
    text = _NOTES_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


class TranscriptPreprocessor:
    """Cleans raw subtitle cues and merges them into utterances.

    Merging rules:
    * consecutive cues are joined while the resulting utterance is shorter than
      ``max_chars`` and the pause between cues is at most ``max_pause``;
    * consecutive duplicate cues (an auto-subtitle artifact) are collapsed;
    * a cue ending with sentence punctuation closes the utterance early only if
      the utterance is already long enough (``min_chars``).
    """

    def __init__(
        self,
        max_chars: int = 600,
        max_pause: float = 2.0,
        min_chars: int = 80,
    ) -> None:
        self._max_chars = max_chars
        self._max_pause = max_pause
        self._min_chars = min_chars

    def prepare(self, transcript: Transcript) -> PreparedTranscript:
        cleaned: list[tuple[float, float, str]] = []
        for entry in transcript.entries:
            text = _clean_text(entry.text)
            if not text:
                continue
            if cleaned and cleaned[-1][2] == text and abs(cleaned[-1][1] - entry.start) < 30.0:
                # Auto-subs often duplicate the same cue over its full duration.
                cleaned[-1] = (cleaned[-1][0], max(cleaned[-1][1], entry.end), text)
                continue
            cleaned.append((entry.start, entry.end, text))
        if not cleaned:
            raise ValueError("После очистки транскрипция пуста")

        merged: list[PreparedEntry] = []
        cur_start, cur_end, cur_parts = cleaned[0][0], cleaned[0][1], [cleaned[0][2]]
        for start, end, text in cleaned[1:]:
            pause = start - cur_end
            candidate = " ".join(cur_parts + [text])
            ends_sentence = cur_parts[-1].rstrip().endswith((".", "!", "?"))
            long_enough = len(" ".join(cur_parts)) >= self._min_chars
            should_close = ends_sentence and long_enough
            if pause > self._max_pause or len(candidate) > self._max_chars or should_close:
                merged.append(PreparedEntry(start=cur_start, end=cur_end, text=" ".join(cur_parts)))
                cur_start, cur_parts = start, [text]
            else:
                cur_parts.append(text)
            cur_end = end
        merged.append(PreparedEntry(start=cur_start, end=cur_end, text=" ".join(cur_parts)))

        result = PreparedTranscript(language=transcript.language, entries=merged)
        logger.info(
            "Подготовка транскрипции: %d реплик -> %d объединённых фрагментов",
            len(transcript.entries),
            len(merged),
        )
        return result
