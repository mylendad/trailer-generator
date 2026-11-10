from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TrailerSegment(BaseModel):
    """One selected source interval placed into the trailer."""

    order: int = Field(ge=1)
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    purpose: str = Field(min_length=1)
    main_idea: str = Field(min_length=1)
    reason: str = ""
    topics: list[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class TrailerScript(BaseModel):
    """Final trailer scenario: an ordered sequence of source time intervals."""

    target_duration: float = Field(gt=0.0)
    segments: list[TrailerSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_orders(self) -> TrailerScript:
        orders = [s.order for s in self.segments]
        expected = list(range(1, len(orders) + 1))
        if sorted(orders) != expected:
            raise ValueError(f"segment orders must be 1..N without gaps, got {orders}")
        for seg in self.segments:
            if seg.end <= seg.start:
                raise ValueError(f"segment {seg.order}: end must be greater than start")
        return self

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.segments)
