from __future__ import annotations

from trailer.analysis.segmenter import TranscriptSegmenter
from trailer.config import BlockSettings
from trailer.domain.models import PreparedEntry, PreparedTranscript
from tests.conftest import make_raw_transcript
from trailer.analysis.transcript_preprocessor import TranscriptPreprocessor


def _prepared() -> PreparedTranscript:
    raw = make_raw_transcript(total_seconds=600.0)
    return TranscriptPreprocessor().prepare(raw)


class TestSegmenter:
    def test_blocks_cover_full_timespan(self):
        prepared = _prepared()
        blocks = TranscriptSegmenter(BlockSettings()).segment(prepared)
        assert blocks[0].start <= prepared.entries[0].start + 1e-6
        assert blocks[-1].end >= prepared.entries[-1].end - 1e-6
        for prev, cur in zip(blocks, blocks[1:], strict=False):
            assert prev.start < prev.end
            assert cur.start >= prev.start  # monotonic

    def test_respects_char_limit(self):
        prepared = _prepared()
        blocks = TranscriptSegmenter(
            BlockSettings(max_block_chars=500, max_block_seconds=10_000.0)
        ).segment(prepared)
        assert len(blocks) > 1
        for block in blocks:
            assert len(block.text) <= 500 + 400  # a single long sentence may exceed slightly

    def test_respects_seconds_limit(self):
        prepared = _prepared()
        blocks = TranscriptSegmenter(
            BlockSettings(max_block_chars=1_000_000, max_block_seconds=120.0)
        ).segment(prepared)
        for block in blocks:
            assert block.end - block.start <= 120.0 + 10.0

    def test_block_ids_unique_and_sequential(self):
        prepared = _prepared()
        blocks = TranscriptSegmenter(BlockSettings()).segment(prepared)
        ids = [block.block_id for block in blocks]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)

    def test_no_lost_text(self):
        prepared = _prepared()
        blocks = TranscriptSegmenter(BlockSettings()).segment(prepared)
        joined = " ".join(block.text for block in blocks)
        for word in prepared.entries[0].text.split():
            assert word in joined

    def test_pause_breaks_block(self):
        entries = [
            PreparedEntry(start=0.0, end=8.0, text="Первая часть. Здесь несколько предложений."),
            PreparedEntry(start=30.0, end=38.0, text="Вторая часть. Другая тема разговора."),
        ]
        prepared = PreparedTranscript(language="ru", entries=entries)
        blocks = TranscriptSegmenter(BlockSettings(max_pause_inside_block=4.0)).segment(prepared)
        assert len(blocks) == 2
        assert blocks[0].end <= 8.0
        assert blocks[1].start >= 30.0
