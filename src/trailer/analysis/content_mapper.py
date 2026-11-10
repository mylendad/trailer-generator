from __future__ import annotations

import logging

from trailer.analysis.block_analyzer import render_analyses_for_prompt
from trailer.analysis.llm.base import LLMClient
from trailer.analysis.prompts import load_prompt
from trailer.domain.models import BlockAnalysis, ContentMap

logger = logging.getLogger(__name__)

_STAGE = "map"


class ContentMapper:
    """Stage B: builds the overall content map of the whole talk."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._system_prompt = load_prompt("build_content_map")

    def build_map(self, analyses: list[BlockAnalysis]) -> ContentMap:
        user_prompt = (
            "Результаты анализа блоков лекции (хронологический порядок):\n\n"
            f"{render_analyses_for_prompt(analyses)}"
        )
        content_map = self._llm.complete_json(self._system_prompt, user_prompt, ContentMap)
        logger.info(
            "Карта содержания: тема=%r, подтем=%d, идей=%d",
            content_map.main_topic,
            len(content_map.subtopics),
            len(content_map.key_ideas),
        )
        return content_map
