from __future__ import annotations

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    """Metadata of the source video, independent of where it came from."""

    video_id: str
    title: str
    duration: float = Field(ge=0.0)
    url: str

    @property
    def duration_mmss(self) -> str:
        minutes, seconds = divmod(int(self.duration), 60)
        return f"{minutes:02d}:{seconds:02d}"
