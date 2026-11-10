from __future__ import annotations

import pytest

from trailer.config import Settings
from trailer.domain.exceptions import ScriptValidationError
from trailer.domain.models import (
    PreparedEntry,
    PreparedTranscript,
    Transcript,
    TranscriptEntry,
    TrailerScript,
)
from trailer.validation.trailer_validator import TrailerValidator


def _prepared() -> PreparedTranscript:
    entries = []
    t = 0.0
    for i in range(40):
        entries.append(
            PreparedEntry(
                start=t, end=t + 8.0, text=f"Полный самостоятельный фрагмент номер {i} с идеей."
            )
        )
        t += 10.0
    return PreparedTranscript(language="ru", entries=entries)


def _script(segments: list[dict], target: float = 50.0) -> TrailerScript:
    return TrailerScript.model_validate({"target_duration": target, "segments": segments})


def _validator(settings: Settings) -> TrailerValidator:
    return TrailerValidator(settings, _prepared(), video_duration=400.0)


def _seg(order: int, start: float, end: float) -> dict:
    return {"order": order, "start": start, "end": end, "purpose": "key_idea", "main_idea": "идея"}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        target_duration=50.0,
        duration_tolerance=10.0,
        workspace_dir=tmp_path,
    )


class TestTrailerValidator:
    def test_accepts_valid_script_and_keeps_boundaries(self, settings):
        validator = _validator(settings)
        script = _script([_seg(1, 5.0, 20.0), _seg(2, 45.0, 60.0), _seg(3, 105.0, 125.0)])
        result = validator.validate(script)
        assert len(result.kept) == 3
        assert result.kept[0].start == pytest.approx(5.0)
        assert result.kept[0].end == pytest.approx(20.0)
        assert 40.0 <= result.total_duration <= 60.0
        assert result.kept[0].order == 1

    def test_snaps_borders_only_within_padding(self, tmp_path):
        settings = Settings(target_duration=33.0, duration_tolerance=5.0, workspace_dir=tmp_path)
        validator = TrailerValidator(settings, _prepared(), video_duration=400.0)
        script = _script([_seg(1, 0.2, 20.0), _seg(2, 45.0, 57.9)], target=33.0)
        result = validator.validate(script)
        assert result.kept[0].start == pytest.approx(0.0)
        assert result.kept[0].end == pytest.approx(20.0)
        assert result.kept[1].start == pytest.approx(45.0)
        assert result.kept[1].end == pytest.approx(58.4)

    def test_rejects_interval_outside_video(self, settings):
        validator = _validator(settings)
        script = _script([_seg(1, 500.0, 520.0)])
        with pytest.raises(ScriptValidationError):
            validator.validate(script)

    def test_rejects_unbacked_interval(self, settings):
        validator = _validator(settings)
        # A gap between utterances (e.g. 8..10 has partial overlap; use pure gap 395..399)
        script = _script([_seg(1, 399.0, 399.5)])
        with pytest.raises(ScriptValidationError, match="отклонены"):
            validator.validate(script)

    def test_rejects_duplicate_intervals(self, settings):
        validator = _validator(settings)
        script = _script([_seg(1, 5.0, 15.0), _seg(2, 12.0, 16.0)])
        with pytest.raises(ScriptValidationError):
            validator.validate(script)

    def test_rejects_total_duration_out_of_range(self, settings):
        validator = _validator(settings)
        script = _script([_seg(1, 5.0, 15.0)])
        with pytest.raises(ScriptValidationError, match="вне допустимого диапазона"):
            validator.validate(script)

    def test_clamps_padding_to_video_bounds(self, settings):
        validator = _validator(settings)
        script = _script([_seg(1, 390.0, 400.0), _seg(2, 200.0, 215.0), _seg(3, 300.0, 315.0)])
        result = validator.validate(script)
        assert all(seg.end <= 400.0 for seg in result.kept)
        assert all(seg.start >= 0.0 for seg in result.kept)

    def test_orders_are_renumbered_after_rejection(self, settings):
        validator = _validator(settings)
        script = _script([_seg(1, 399.0, 399.5), _seg(2, 5.0, 15.0), _seg(3, 200.0, 215.0)])
        with pytest.raises(ScriptValidationError, match="вне допустимого диапазона"):
            validator.validate(script)

    def test_aligns_segment_borders_to_raw_cues(self, tmp_path):
        settings = Settings(target_duration=30.0, duration_tolerance=60.0, workspace_dir=tmp_path)
        raw = Transcript(
            language="ru",
            entries=[
                TranscriptEntry(start=0.0, duration=5.0, text="раз"),
                TranscriptEntry(start=5.0, duration=7.0, text="два"),
                TranscriptEntry(start=12.0, duration=7.0, text="три"),
                TranscriptEntry(start=19.0, duration=7.0, text="четыре"),
                TranscriptEntry(start=26.0, duration=7.0, text="пять"),
            ],
        )
        validator = TrailerValidator(settings, _prepared(), 400.0, raw)
        script = _script([_seg(1, 10.0, 20.0)], target=30.0)
        result = validator.validate(script)
        assert result.kept[0].start == pytest.approx(5.0)
        assert result.kept[0].end == pytest.approx(26.0)
