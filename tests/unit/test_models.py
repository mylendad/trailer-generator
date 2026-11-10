from __future__ import annotations

import pytest
from pydantic import ValidationError

from trailer.domain.models import (
    PreparedEntry,
    SegmentCandidate,
    TrailerScript,
    TrailerSegment,
    Transcript,
    TranscriptEntry,
    VideoInfo,
)


class TestModels:
    def test_video_info_mmss(self):
        info = VideoInfo(video_id="x", title="t", duration=3725.0, url="u")
        assert info.duration_mmss == "62:05"

    def test_transcript_entry_requires_text(self):
        with pytest.raises(ValidationError):
            TranscriptEntry(start=1.0, duration=2.0, text="")

    def test_prepared_entry_rejects_inverted_interval(self):
        with pytest.raises(ValidationError):
            PreparedEntry(start=10.0, end=10.0, text="текст")

    def test_candidate_value_bounds(self):
        with pytest.raises(ValidationError):
            SegmentCandidate(start=0, end=10, main_idea="x", value=11)

    def test_script_rejects_duplicate_orders(self):
        with pytest.raises(ValidationError):
            TrailerScript.model_validate(
                {
                    "target_duration": 90.0,
                    "segments": [
                        {"order": 1, "start": 0.0, "end": 5.0, "purpose": "hook", "main_idea": "a"},
                        {
                            "order": 1,
                            "start": 10.0,
                            "end": 15.0,
                            "purpose": "hook",
                            "main_idea": "b",
                        },
                    ],
                }
            )

    def test_script_total_duration(self):
        script = TrailerScript.model_validate(
            {
                "target_duration": 90.0,
                "segments": [
                    {"order": 1, "start": 0.0, "end": 10.0, "purpose": "hook", "main_idea": "a"},
                    {
                        "order": 2,
                        "start": 20.0,
                        "end": 35.0,
                        "purpose": "key_idea",
                        "main_idea": "b",
                    },
                ],
            }
        )
        assert script.total_duration == 25.0

    def test_transcript_requires_entries(self):
        with pytest.raises(ValidationError):
            Transcript(language="ru", entries=[])

    def test_trailer_segment_duration(self):
        seg = TrailerSegment(order=1, start=5.0, end=12.5, purpose="hook", main_idea="x")
        assert seg.duration == 7.5
