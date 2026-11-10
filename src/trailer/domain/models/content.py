from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentCandidate(BaseModel):
    """A time interval proposed by the LLM as potentially valuable for the trailer."""

    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    main_idea: str = Field(min_length=1)
    context: str = ""
    value: int = Field(ge=1, le=10)
    self_contained: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


class BlockAnalysis(BaseModel):
    """Result of the LLM analysis of a single transcript block (Stage A)."""

    block_id: str
    summary: str
    topics: list[str] = Field(default_factory=list)
    key_ideas: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    has_complete_thought: bool = False
    candidates: list[SegmentCandidate] = Field(default_factory=list)


class Subtopic(BaseModel):
    title: str
    description: str
    covered_by_blocks: list[str] = Field(default_factory=list)


class ContentMap(BaseModel):
    """Overall content map of the whole talk (Stage B)."""

    main_topic: str
    narrative_flow: list[str] = Field(default_factory=list)
    subtopics: list[Subtopic] = Field(default_factory=list)
    key_problems: list[str] = Field(default_factory=list)
    key_ideas: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
