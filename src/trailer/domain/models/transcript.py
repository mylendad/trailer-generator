from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TranscriptEntry(BaseModel):
    """A single raw subtitle cue with its timing, as delivered by the provider."""

    start: float = Field(ge=0.0)
    duration: float = Field(gt=0.0)
    text: str = Field(min_length=1)

    @property
    def end(self) -> float:
        return self.start + self.duration


class Transcript(BaseModel):
    """Raw transcript: an ordered list of cue entries with timings preserved."""

    language: str
    entries: list[TranscriptEntry] = Field(min_length=1)

    @property
    def start(self) -> float:
        return self.entries[0].start

    @property
    def end(self) -> float:
        return max(e.end for e in self.entries)

    @property
    def text(self) -> str:
        return " ".join(e.text.strip() for e in self.entries)


class PreparedEntry(BaseModel):
    """A merged, cleaned utterance: several consecutive cues joined into one."""

    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_order(self) -> PreparedEntry:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self


class PreparedTranscript(BaseModel):
    """Cleaned transcript prepared for LLM analysis; timings preserved."""

    language: str
    entries: list[PreparedEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_sorted(self) -> PreparedTranscript:
        for prev, cur in zip(self.entries, self.entries[1:], strict=False):
            if cur.start < prev.start - 1e-9:
                raise ValueError("prepared entries must be sorted by start time")
        return self

    @property
    def end(self) -> float:
        return self.entries[-1].end

    def text_at(self, start: float, end: float) -> str:
        """Return the prepared text that falls inside [start, end]."""
        chunks = [e.text for e in self.entries if e.start < end - 1e-9 and e.end > start + 1e-9]
        return " ".join(chunks)


class TranscriptBlock(BaseModel):
    """A semantically coherent chunk of the transcript, suitable for one LLM call."""

    block_id: str
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_order(self) -> TranscriptBlock:
        if self.end <= self.start:
            raise ValueError("block end must be greater than start")
        return self
