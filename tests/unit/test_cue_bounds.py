from __future__ import annotations

import pytest

from trailer.domain.models import Transcript, TranscriptEntry
from trailer.transcription.cue_bounds import CueBounds

# Mirrors the real rolling-subtitles pattern around the faulty cut:
# consecutive cues overlap, so a naive nearest-boundary snap lands mid-word.
_SPANS = [
    (3391.05, 3395.58),  # опять-таки после изучения у каждой тема
    (3394.11, 3397.40),  # ребенок уже делать свой собственный
    (3395.58, 3400.38),  # проект и
    (3397.40, 3403.35),  # то что касается эта информация по
    (3400.38, 3407.04),  # примеру втором классе вы видите заложено
    (3428.34, 3434.64),  # грамотности и еще один вариант
    (3432.02, 3436.28),  # пособие внеурочной деятельности мир
    (3434.64, 3439.68),  # вокруг нас опыты эксперементы
    (3436.28, 3440.99),  # практические задания которые помогают
    (3439.68, 3442.91),  # детям стало действительно
    (3440.99, 3447.90),  # исследователями
    (3442.91, 3450.68),  # больше узнать об окружающем мире в
    (3447.90, 3453.72),  # первом классе мы работаем со стихиями
]


def _bounds() -> CueBounds:
    return CueBounds(
        starts=sorted({s for s, _ in _SPANS}),
        ends=sorted({e for _, e in _SPANS}),
        spans=list(_SPANS),
    )


def _transcript() -> Transcript:
    return Transcript(
        language="ru",
        entries=[
            TranscriptEntry(start=start, duration=end - start, text=f"cue {index}")
            for index, (start, end) in enumerate(_SPANS)
        ],
    )


class TestSnapStart:
    def test_prefers_enclosing_cue_start_over_neighbor(self):
        # 3400.28 sits inside cue 3395.58-3400.38 AND 3397.40-3403.35;
        # the latest enclosing start is 3397.40.
        assert _bounds().snap_start(3400.28) == pytest.approx(3397.40)

    def test_exact_start_is_kept(self):
        assert _bounds().snap_start(3434.64) == pytest.approx(3434.64)

    def test_falls_back_to_nearest_start_before(self):
        assert _bounds().snap_start(9999.0) == pytest.approx(3447.90)


class TestSnapEnd:
    def test_end_snaps_to_enclosing_cue_not_neighbor(self):
        # 3435.28 sits inside cue 3434.64-3439.68 ("вокруг нас опыты
        # эксперементы"). The buggy nearest-boundary snap picked 3436.28 —
        # the end of the *neighbouring* cue — and cut mid-word.
        assert _bounds().snap_end(3435.28) == pytest.approx(3439.68)

    def test_max_enclosing_end_wins(self):
        # 3436.5 sits inside 3434.64-3439.68 and 3436.28-3440.99;
        # the latest enclosing end is 3440.99.
        assert _bounds().snap_end(3436.5) == pytest.approx(3440.99)

    def test_value_in_gap_uses_last_end_before(self):
        assert _bounds().snap_end(9999.0) == pytest.approx(3453.72)


class TestFromTranscript:
    def test_builds_from_transcript(self):
        bounds = CueBounds.from_transcript(_transcript())
        assert bounds is not None
        assert bounds.snap_end(3435.28) == pytest.approx(3439.68)

    def test_none_transcript_yields_none(self):
        assert CueBounds.from_transcript(None) is None
