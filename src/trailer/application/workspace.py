"""Centralised workspace layout for one processed video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Workspace:
    """All paths of a single video's working directory."""

    root: Path

    @property
    def video_info(self) -> Path:
        return self.root / "video_info.json"

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def original_video(self) -> Path:
        return self.source_dir / "original.mp4"

    @property
    def transcript_dir(self) -> Path:
        return self.root / "transcript"

    @property
    def raw_transcript(self) -> Path:
        return self.transcript_dir / "raw.json"

    @property
    def prepared_transcript(self) -> Path:
        return self.transcript_dir / "prepared.json"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def blocks(self) -> Path:
        return self.analysis_dir / "blocks.json"

    @property
    def block_analysis(self) -> Path:
        return self.analysis_dir / "block_analysis.json"

    @property
    def content_map(self) -> Path:
        return self.analysis_dir / "content_map.json"

    @property
    def candidates(self) -> Path:
        return self.analysis_dir / "candidates.json"

    @property
    def trailer_dir(self) -> Path:
        return self.root / "trailer"

    @property
    def script(self) -> Path:
        return self.trailer_dir / "script.json"

    @property
    def validated_script(self) -> Path:
        return self.trailer_dir / "script_validated.json"

    @property
    def validation_report(self) -> Path:
        return self.trailer_dir / "validation_report.json"

    @property
    def trailer_segments_dir(self) -> Path:
        return self.trailer_dir / "segments"

    @property
    def trailer_video(self) -> Path:
        return self.trailer_dir / "trailer.mp4"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "pipeline.log"

    def ensure_directories(self) -> None:
        for directory in (
            self.root,
            self.source_dir,
            self.transcript_dir,
            self.analysis_dir,
            self.trailer_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
