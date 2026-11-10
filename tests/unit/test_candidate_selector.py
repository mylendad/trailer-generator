from __future__ import annotations

from trailer.analysis.candidate_selector import CandidateSelector
from trailer.config import SelectionSettings
from trailer.domain.models import BlockAnalysis, SegmentCandidate, Transcript, TranscriptEntry


def _analysis(*candidates: SegmentCandidate) -> BlockAnalysis:
    return BlockAnalysis(
        block_id="block_001",
        summary="s",
        candidates=list(candidates),
    )


def _candidate(start: float, end: float, idea: str, value: int = 8) -> SegmentCandidate:
    return SegmentCandidate(
        start=start,
        end=end,
        main_idea=idea,
        context="пример",
        value=value,
        self_contained=True,
    )


class TestCandidateSelector:
    def test_drops_non_self_contained_and_out_of_range(self):
        selector = CandidateSelector(SelectionSettings(), 90.0)
        result = selector.select(
            [
                _analysis(
                    _candidate(10, 30, "Хорошая идея"),
                    _candidate(100, 104, "Слишком короткий"),
                    _candidate(200, 260, "Слишком длинный"),
                    SegmentCandidate(
                        start=300,
                        end=320,
                        main_idea="Не самостоятельный",
                        value=10,
                        self_contained=False,
                    ),
                )
            ]
        )
        assert [c.main_idea for c in result.accepted] == ["Хорошая идея"]
        assert len(result.reasons) == 3

    def test_deduplicates_similar_ideas(self):
        selector = CandidateSelector(SelectionSettings(), 90.0)
        result = selector.select(
            [
                _analysis(
                    _candidate(10, 30, "Традиционный подход не масштабируется"),
                    _candidate(100, 120, "Традиционный подход не масштабируется при росте"),
                    _candidate(200, 220, "Пример из практики микросервисов"),
                )
            ]
        )
        ideas = [c.main_idea for c in result.accepted]
        assert "Пример из практики микросервисов" in ideas
        assert len(ideas) == 2

    def test_pool_is_capped_by_duration_budget(self):
        selector = CandidateSelector(
            SelectionSettings(candidate_pool_factor=1.0, dedupe_similarity=0.99),
            target_duration=90.0,
        )
        result = selector.select(
            [
                _analysis(
                    *[
                        _candidate(10 + i * 30, 30 + i * 30, f"Идея номер {i}", value=9)
                        for i in range(10)
                    ]
                )
            ]
        )
        total = sum(c.duration for c in result.accepted)
        assert 90.0 <= total < 90.0 + 20.0  # stops once the budget is reached

    def test_higher_value_ranks_first(self):
        selector = CandidateSelector(SelectionSettings(), 90.0)
        result = selector.select(
            [
                _analysis(
                    _candidate(10, 25, "Обычная идея", value=3),
                    _candidate(100, 115, "Важная идея", value=10),
                )
            ]
        )
        assert result.accepted[0].main_idea == "Важная идея"

    def test_snaps_candidate_bounds_to_transcript_cues(self):
        transcript = Transcript(
            language="ru",
            entries=[
                TranscriptEntry(start=0.0, duration=6.0, text="раз"),
                TranscriptEntry(start=6.0, duration=7.0, text="два"),
                TranscriptEntry(start=13.0, duration=6.0, text="три"),
                TranscriptEntry(start=19.0, duration=6.0, text="четыре"),
            ],
        )
        selector = CandidateSelector(SelectionSettings(), 90.0, transcript)
        result = selector.select([_analysis(_candidate(10, 20, "Идея"))])
        assert result.accepted[0].start == 6.0
        assert result.accepted[0].end == 25.0
