from __future__ import annotations

import json
import logging

from trailer.analysis.llm.base import LLMClient
from trailer.analysis.prompts import load_prompt
from trailer.config import Settings
from trailer.domain.models import ContentMap, SegmentCandidate, TrailerScript, TrailerSegment

logger = logging.getLogger(__name__)

_STAGE = "plan"


class TrailerPlanner:
    """Stage C: turns the content map plus the candidate pool into a script."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings
        self._system_prompt = load_prompt("create_trailer")

    def build_script(
        self,
        content_map: ContentMap,
        candidates: list[SegmentCandidate],
        exclude: list[TrailerSegment] | None = None,
    ) -> TrailerScript:
        if not candidates:
            raise ValueError("Нет кандидатов для формирования сценария")
        system_prompt = self._system_prompt.format(
            target_duration=self._settings.target_duration,
            tolerance=self._settings.duration_tolerance,
        )
        user_prompt = (
            "Карта содержания лекции:\n"
            f"{json.dumps(content_map.model_dump(mode='json'), ensure_ascii=False, indent=1)}\n\n"
            "Проверенные кандидаты (используй только эти таймкоды):\n"
            f"{json.dumps([c.model_dump(mode='json') for c in candidates], ensure_ascii=False, indent=1)}\n\n"
            f"Целевая длительность трейлера: {self._settings.target_duration} секунд "
            f"(допустимо ±{self._settings.duration_tolerance} секунд)."
        )
        if exclude:
            user_prompt += (
                "\n\nУже использованные фрагменты (это другой вариант трейлера того же "
                "видео; НЕ выбирайте эти интервалы и близкие к ним по смыслу идеи):\n"
                f"{json.dumps([s.model_dump(mode='json') for s in exclude], ensure_ascii=False, indent=1)}"
            )
        script = self._llm.complete_json(system_prompt, user_prompt, TrailerScript)
        script = self._fit_to_target(script)
        logger.info(
            "Сценарий: %d фрагментов, суммарная длительность %.1f с",
            len(script.segments),
            script.total_duration,
        )
        return script

    def _fit_to_target(self, script: TrailerScript) -> TrailerScript:
        """Deterministically trim an over-long script to the allowed window.

        The hook and the conclusion are preserved; middle segments are kept in
        narrative order while they fit into the remaining budget.
        """
        if script.total_duration <= self._settings.duration_max:
            return script
        segments = sorted(script.segments, key=lambda s: s.order)
        if len(segments) <= 2:
            return script
        head, tail = segments[0], segments[-1]
        budget = self._settings.duration_max - head.duration - tail.duration
        kept_middle = []
        for candidate in segments[1:-1]:
            if candidate.duration <= budget:
                kept_middle.append(candidate)
                budget -= candidate.duration
        fitted = [head, *kept_middle, tail]
        if len(fitted) == len(segments):
            return script
        logger.warning(
            "Сценарий сокращён под целевую длительность: %d -> %d фрагментов",
            len(segments),
            len(fitted),
        )
        return TrailerScript(
            target_duration=script.target_duration,
            segments=[
                segment.model_copy(update={"order": index})
                for index, segment in enumerate(fitted, start=1)
            ],
        )
