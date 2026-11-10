from __future__ import annotations

import pytest
from pydantic import ValidationError

from trailer.analysis.transcript_preprocessor import TranscriptPreprocessor
from tests.conftest import make_raw_transcript

from trailer.domain.models import (
    Transcript,
    TranscriptEntry,
)


def _transcript(*texts: str, start: float = 0.0, duration: float = 2.0) -> Transcript:
    entries = [
        TranscriptEntry(start=start + i * duration, duration=duration, text=text)
        for i, text in enumerate(texts)
    ]
    return Transcript(language="ru", entries=entries)


class TestCleaning:
    def test_removes_music_tags(self):
        transcript = _transcript(
            "[Музыка]", "Сегодня мы поговорим о масштабировании.", "(аплодисменты)"
        )
        prepared = TranscriptPreprocessor().prepare(transcript)
        assert "Сегодня мы поговорим о масштабировании." in prepared.entries[0].text
        assert "Музыка" not in prepared.entries[0].text

    def test_removes_notes_symbols(self):
        transcript = _transcript("♪♪", "Итак начнём >> с главного.", "♪")
        prepared = TranscriptPreprocessor().prepare(transcript)
        assert "♪" not in prepared.entries[0].text
        assert ">>" not in prepared.entries[0].text

    def test_collapses_auto_subtitle_duplicates(self):
        transcript = _transcript(
            "автоматические субтитры часто дублируются",
            "автоматические субтитры часто дублируются",
            "автоматические субтитры часто дублируются",
        )
        prepared = TranscriptPreprocessor().prepare(transcript)
        assert len(prepared.entries) == 1
        assert "дублируются" in prepared.entries[0].text

    def test_all_artifacts_only_raises(self):
        transcript = _transcript("[Музыка]", "(смех)")
        with pytest.raises(ValueError, match="пуста"):
            TranscriptPreprocessor().prepare(transcript)


class TestMerging:
    def test_merges_cues_until_pause(self):
        texts = [f"слова номер {i}" for i in range(5)]
        transcript = _transcript(*texts)
        prepared = TranscriptPreprocessor(max_pause=1.0).prepare(transcript)
        assert all(entry.start < entry.end for entry in prepared.entries)
        # Everything is merged into one utterance when there is no real pause.
        assert len(prepared.entries) == 1
        assert prepared.entries[0].text.count("слова") == 5

    def test_splits_on_long_pause(self):
        texts = ["первая мысль", "вторая мысль", "третья мысль"]
        transcript = _transcript(*texts, duration=2.0)
        # Gap of 2s between cues exceeds max_pause=1.0? No: entries are
        # contiguous (duration == step), so craft a real gap.
        entries = [
            TranscriptEntry(start=0.0, duration=1.0, text="первая мысль"),
            TranscriptEntry(start=5.0, duration=1.0, text="вторая мысль"),
        ]
        transcript = Transcript(language="ru", entries=entries)
        prepared = TranscriptPreprocessor(max_pause=2.0).prepare(transcript)
        assert len(prepared.entries) == 2
        assert prepared.entries[0].text == "первая мысль"
        assert prepared.entries[1].start == 5.0

    def test_preserves_start_of_first_and_end_of_last(self):
        transcript = make_raw_transcript(total_seconds=30.0)
        prepared = TranscriptPreprocessor().prepare(transcript)
        assert prepared.entries[0].start == transcript.entries[0].start
        last = transcript.entries[-1]
        assert prepared.entries[-1].end == last.start + last.duration

    def test_max_chars_respected(self):
        texts = [f"очень длинная реплика номер {i} с дополнительными словами" for i in range(10)]
        transcript = _transcript(*texts)
        prepared = TranscriptPreprocessor(max_chars=120).prepare(transcript)
        assert all(len(entry.text) <= 160 for entry in prepared.entries)  # one cue may exceed


class TestValidation:
    def test_sorted_entries_enforced(self):
        from trailer.domain.models import PreparedEntry, PreparedTranscript

        with pytest.raises(ValidationError):
            PreparedTranscript(
                language="ru",
                entries=[
                    PreparedEntry(start=10.0, end=12.0, text="а"),
                    PreparedEntry(start=0.0, end=5.0, text="б"),
                ],
            )
