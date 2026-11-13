from __future__ import annotations

import logging
from dataclasses import dataclass, field

from trailer.config import Settings
from trailer.domain.exceptions import ScriptValidationError
from trailer.domain.models import (
    PreparedTranscript,
    Transcript,
    TrailerScript,
    TrailerSegment,
)
from trailer.transcription.cue_bounds import CueBounds

logger = logging.getLogger(__name__)

_STAGE = "validate"


@dataclass(slots=True)
class ValidationResult:
    kept: list[TrailerSegment] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.kept)


class TrailerValidator:
    """Checks an LLM-produced script against the source data before editing.

    Every interval must be inside the video, must be backed by real transcript
    utterances, and must not duplicate another interval. The total duration
    must not fall below the configured minimum; when it exceeds the hard cap
    (``max_allowed_duration``), trailing fragments are dropped instead of
    failing. Intervals are snapped to utterance boundaries with a small
    padding so speech is not clipped mid-word.
    """

    def __init__(
        self,
        settings: Settings,
        prepared: PreparedTranscript,
        video_duration: float,
        raw: Transcript | None = None,
    ) -> None:
        self._settings = settings
        self._prepared = prepared
        self._video_duration = video_duration
        self._bounds = CueBounds.from_transcript(raw)

    def validate(self, script: TrailerScript) -> ValidationResult:
        result = ValidationResult()
        seen_ranges: list[tuple[float, float]] = []

        for original in sorted(script.segments, key=lambda s: s.order):
            segment = self._align_to_cues(original)
            issues = self._check_interval(segment)
            if issues:
                result.rejected.append(f"Фрагмент {segment.order}: {issues}")
                continue

            overlapped = self._overlap(segment, seen_ranges)
            if overlapped is not None:
                result.rejected.append(
                    f"Фрагмент {segment.order}: дублирует интервал фрагмента {overlapped}"
                )
                continue

            snapped = self._snap(segment)
            if snapped is None:
                result.rejected.append(
                    f"Фрагмент {segment.order}: интервал не подтверждён транскрипцией"
                )
                continue
            if snapped is not original:
                result.adjustments.append(
                    f"Фрагмент {segment.order}: границы скорректированы "
                    f"{original.start:.1f}-{original.end:.1f} -> {snapped.start:.1f}-{snapped.end:.1f}"
                )
            seen_ranges.append((snapped.start, snapped.end))
            result.kept.append(snapped)

        if not result.kept:
            raise ScriptValidationError(
                _STAGE, "Все фрагменты сценария отклонены: " + "; ".join(result.rejected)
            )

        max_allowed = self._settings.max_allowed_duration
        while result.kept and result.total_duration > max_allowed:
            dropped = result.kept.pop()
            result.rejected.append(
                f"Фрагмент {dropped.order}: убран, итоговая длительность превышает лимит "
                f"{max_allowed:.0f} с"
            )
        if not result.kept:
            raise ScriptValidationError(
                _STAGE, "Все фрагменты сценария отклонены: " + "; ".join(result.rejected)
            )

        result.kept = [
            seg.model_copy(update={"order": i}) for i, seg in enumerate(result.kept, start=1)
        ]

        if result.total_duration < self._settings.duration_min:
            raise ScriptValidationError(
                _STAGE,
                f"Итоговая длительность {result.total_duration:.1f} с вне допустимого диапазона "
                f"меньше минимальной {self._settings.duration_min:.0f} с "
                f"({len(result.kept)} фрагментов)",
            )
        logger.info(
            "Валидация сценария: %d принято, %d отклонено, итоговая длительность %.1f с",
            len(result.kept),
            len(result.rejected),
            result.total_duration,
        )
        return result

    def build_validated_script(
        self, script: TrailerScript
    ) -> tuple[TrailerScript, ValidationResult]:
        result = self.validate(script)
        validated = TrailerScript(
            target_duration=script.target_duration,
            segments=result.kept,
        )
        return validated, result

    def _check_interval(self, segment: TrailerSegment) -> str | None:
        if segment.start < 0:
            return "начало интервала отрицательное"
        if segment.end > self._video_duration + 1.0:
            return (
                f"конец интервала ({segment.end:.1f}) за пределами видео "
                f"({self._video_duration:.1f})"
            )
        return None

    def _overlap(self, segment: TrailerSegment, ranges: list[tuple[float, float]]) -> int | None:
        for index, (start, end) in enumerate(ranges, start=1):
            if segment.start < end - 1e-6 and segment.end > start + 1e-6:
                return index
        return None

    def _align_to_cues(self, segment: TrailerSegment) -> TrailerSegment:
        """Snap borders to the enclosing subtitle cues and speech pauses."""
        if self._bounds is None:
            return segment
        start = self._bounds.snap_start(segment.start)
        end = min(self._bounds.snap_end(segment.end), self._video_duration)
        if end - start < 1.0:
            return segment
        if abs(start - segment.start) < 1e-6 and abs(end - segment.end) < 1e-6:
            return segment
        return segment.model_copy(update={"start": start, "end": end})

    def _snap(self, segment: TrailerSegment) -> TrailerSegment | None:
        """Nudge borders to utterance boundaries only when they are very close.

        Expanding every segment to the full overlapped utterances would balloon
        the trailer duration, so only borders within ``segment_padding`` of an
        entry boundary are aligned.
        """
        padding = self._settings.selection.segment_padding
        inner = [
            e for e in self._prepared.entries if e.start < segment.end and e.end > segment.start
        ]
        if not inner:
            return None
        text_len = len(" ".join(e.text for e in inner))
        if text_len < 20:
            logger.warning(
                "Фрагмент %d: слишком мало текста в интервале (%d символов)",
                segment.order,
                text_len,
            )
        new_start = segment.start
        new_end = segment.end
        start_entry = next((e for e in inner if e.start <= segment.start < e.end), None)
        if start_entry is not None and segment.start - start_entry.start <= padding:
            new_start = max(0.0, start_entry.start - padding)
        end_entry = next((e for e in inner if e.start < segment.end <= e.end), None)
        if end_entry is not None and end_entry.end - segment.end <= padding:
            new_end = min(self._video_duration, end_entry.end + padding)
        if new_end - new_start < 1.0:
            return None
        if abs(new_start - segment.start) < 1e-6 and abs(new_end - segment.end) < 1e-6:
            return segment
        return segment.model_copy(update={"start": new_start, "end": new_end})
