from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from trailer.config import SelectionSettings
from trailer.domain.models import BlockAnalysis, SegmentCandidate, Transcript

logger = logging.getLogger(__name__)

_STAGE = "select"


@dataclass(slots=True)
class SelectionResult:
    accepted: list[SegmentCandidate]
    rejected_count: int
    reasons: list[str]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-zа-яё0-9 ]", " ", text.lower()).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _floor(points: list[float], value: float) -> float:
    index = bisect.bisect_right(points, value + 1e-6) - 1
    return points[index] if index >= 0 else value


def _ceil(points: list[float], value: float) -> float:
    index = bisect.bisect_left(points, value - 1e-6)
    return points[index] if index < len(points) else value


def _align_to_cues(
    candidates: list[SegmentCandidate], transcript: Transcript
) -> tuple[list[SegmentCandidate], int]:
    """Snap LLM timecodes to real cue boundaries so cuts never land mid-cue."""
    starts = sorted({entry.start for entry in transcript.entries})
    ends = sorted({entry.end for entry in transcript.entries})
    if not starts or not ends:
        return candidates, 0
    aligned: list[SegmentCandidate] = []
    changed = 0
    for candidate in candidates:
        start = _floor(starts, candidate.start)
        end = _ceil(ends, candidate.end)
        if end - start < 1.0:
            aligned.append(candidate)
            continue
        if abs(start - candidate.start) > 1e-6 or abs(end - candidate.end) > 1e-6:
            changed += 1
            candidate = candidate.model_copy(update={"start": start, "end": end})
        aligned.append(candidate)
    return aligned, changed


class CandidateSelector:
    """Deterministic scoring, deduplication and pooling of LLM candidates.

    The LLM ``value`` score is only one input: the selector additionally rewards
    self-containedness, duration fit and coverage of distinct topics, and drops
    near-duplicate ideas so the planner never sees repeats.
    """

    def __init__(
        self,
        settings: SelectionSettings,
        target_duration: float,
        transcript: Transcript | None = None,
    ) -> None:
        self._settings = settings
        self._target_duration = target_duration
        self._transcript = transcript

    def select(self, analyses: list[BlockAnalysis]) -> SelectionResult:
        candidates = [c for a in analyses for c in a.candidates]
        if self._transcript is not None:
            candidates, changed = _align_to_cues(candidates, self._transcript)
            logger.info(
                "Границы кандидатов выровнены по репликам: %d из %d", changed, len(candidates)
            )
        reasons: list[str] = []

        accepted: list[SegmentCandidate] = []
        for candidate in candidates:
            reason = _reject_reason(candidate, self._settings)
            if reason is None:
                accepted.append(candidate)
            else:
                reasons.append(f"[{candidate.start:.1f}-{candidate.end:.1f}] {reason}")

        accepted.sort(key=lambda c: _score(c, self._settings), reverse=True)

        deduped: list[SegmentCandidate] = []
        for candidate in accepted:
            if any(
                _similarity(candidate.main_idea, kept.main_idea) >= self._settings.dedupe_similarity
                for kept in deduped
            ):
                reasons.append(
                    f"[{candidate.start:.1f}-{candidate.end:.1f}] дубликат уже отобранной идеи"
                )
                continue
            deduped.append(candidate)

        pool_duration = self._target_duration * self._settings.candidate_pool_factor
        pool: list[SegmentCandidate] = []
        total = 0.0
        for candidate in deduped:
            pool.append(candidate)
            total += candidate.duration
            if total >= pool_duration:
                break

        rejected_count = len(candidates) - len(pool)
        logger.info(
            "Отбор кандидатов: %d исходных, %d в пуле (итоговый бюджет %.0f с)",
            len(candidates),
            len(pool),
            pool_duration,
        )
        return SelectionResult(accepted=pool, rejected_count=rejected_count, reasons=reasons)


def _reject_reason(candidate: SegmentCandidate, settings: SelectionSettings) -> str | None:
    if not candidate.self_contained:
        return "фрагмент не самостоятелен"
    if candidate.duration < settings.min_candidate_duration:
        return "слишком короткий"
    if candidate.duration > settings.max_candidate_duration:
        return "слишком длинный"
    return None


def _score(candidate: SegmentCandidate, settings: SelectionSettings) -> float:
    score = float(candidate.value)
    if candidate.self_contained:
        score += 2.0
    middle_duration = (settings.min_candidate_duration + settings.max_candidate_duration) / 2
    score -= abs(candidate.duration - middle_duration) / 10.0
    if candidate.context:
        score += 0.5
    return score
