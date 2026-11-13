"""Boundary snapping against raw subtitle cues (including overlapping ones)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from trailer.domain.models import Transcript


@dataclass(frozen=True, slots=True)
class CueBounds:
    """Snaps timecodes to the cue that *contains* the value.

    Auto-generated subtitles roll: consecutive cues overlap, so the nearest
    cue end may belong to a *neighbouring* cue and land mid-word. The border
    is therefore resolved against the cue containing the value. Final speech
    pauses are detected from the actual audio by the editor.
    """

    starts: list[float]
    ends: list[float]
    spans: list[tuple[float, float]]

    @classmethod
    def from_transcript(cls, transcript: Optional[Transcript]) -> Optional[CueBounds]:
        if transcript is None:
            return None
        spans = sorted((e.start, e.end) for e in transcript.entries)
        if not spans:
            return None
        starts = sorted({start for start, _ in spans})
        ends = sorted({end for _, end in spans})
        return cls(starts, ends, spans)

    def snap_start(self, value: float) -> float:
        """Latest cue start among the cues containing ``value``."""
        containing = [start for start, end in self.spans if start <= value <= end]
        if containing:
            return max(containing)
        before = [start for start in self.starts if start <= value]
        return before[-1] if before else value

    def snap_end(self, value: float) -> float:
        """Latest cue end among the cues containing ``value``."""
        containing = [end for start, end in self.spans if start <= value <= end]
        if containing:
            return max(containing)
        before = [end for end in self.ends if end < value]
        return before[-1] if before else value
