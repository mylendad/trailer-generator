from __future__ import annotations

import json
import logging

from trailer.analysis.llm.base import LLMClient
from trailer.analysis.prompts import load_prompt
from trailer.domain.models import BlockAnalysis, TranscriptBlock

logger = logging.getLogger(__name__)

_STAGE = "analyze"


class BlockAnalyzer:
    """Stage A: per-block content analysis with the LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._system_prompt = load_prompt("analyze_block")

    def analyze_block(self, block: TranscriptBlock) -> BlockAnalysis:
        user_prompt = (
            f"Идентификатор блока: {block.block_id}\n"
            f"Границы блока: от {block.start:.1f} до {block.end:.1f} секунд.\n"
            f"Текст блока:\n{block.text}"
        )
        system_prompt = self._system_prompt.format(block_start=block.start, block_end=block.end)
        analysis = self._llm.complete_json(system_prompt, user_prompt, BlockAnalysis)
        analysis = _clamp_candidates(analysis, block)
        logger.info(
            "Блок %s: %d тем(ы), %d кандидат(ов)",
            block.block_id,
            len(analysis.topics),
            len(analysis.candidates),
        )
        return analysis

    def analyze_all(self, blocks: list[TranscriptBlock]) -> list[BlockAnalysis]:
        results = []
        for index, block in enumerate(blocks, start=1):
            logger.info("Анализ блока %d/%d (%s)", index, len(blocks), block.block_id)
            results.append(self.analyze_block(block))
        return results


def _clamp_candidates(analysis: BlockAnalysis, block: TranscriptBlock) -> BlockAnalysis:
    """Drop candidates whose timecodes the LLM invented or misplaced."""
    kept = [
        candidate
        for candidate in analysis.candidates
        if block.start - 1e-6 <= candidate.start < candidate.end <= block.end + 1e-6
    ]
    if len(kept) != len(analysis.candidates):
        logger.warning(
            "Блок %s: отброшено %d кандидат(ов) с таймкодами вне блока",
            block.block_id,
            len(analysis.candidates) - len(kept),
        )
    analysis.candidates = kept
    return analysis


def render_analyses_for_prompt(analyses: list[BlockAnalysis]) -> str:
    """Compact representation of block analyses for the content-map prompt."""
    return json.dumps(
        [a.model_dump(mode="json") for a in analyses],
        ensure_ascii=False,
        indent=1,
    )
