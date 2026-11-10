from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from tests.conftest import FakeLLM, FakeTranscriptProvider

from trailer.application.pipeline import Pipeline
from trailer.config import BlockSettings, SelectionSettings, Settings
from trailer.domain.models import (
    Transcript,
    TranscriptEntry,
    VideoInfo,
)
from trailer.sources.base import VideoSource

pytestmark = pytest.mark.integration

GRID_STEP = 10.0
GRID_DURATION = 9.5
GRID_COUNT = 20
VIDEO_DURATION = GRID_STEP * GRID_COUNT


def _ffmpeg_path() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        return None


@pytest.fixture(autouse=True)
def _ensure_ffmpeg_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
):
    """Make ffmpeg resolvable for the editor even when not installed globally."""
    exe = _ffmpeg_path()
    if exe is None:
        pytest.skip("ffmpeg не установлен")
    assert exe is not None
    bin_dir = tmp_path_factory.mktemp("ffmpeg_bin")
    # imageio-ffmpeg ships a versioned filename; expose it under the plain name.
    ffmpeg_link = bin_dir / "ffmpeg"
    ffmpeg_link.symlink_to(exe)
    # subprocess.run resolves "ffmpeg" through PATH; expose the binary.
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")


def _grid_transcript() -> Transcript:
    entries = []
    for k in range(GRID_COUNT):
        # A complete sentence of >80 chars ending with a period, so each grid
        # cue stays a standalone utterance after preprocessing.
        text = ("подробное содержательное слово " * 3 + f"фрагмент лекции номер {k}.").strip()
        entries.append(TranscriptEntry(start=k * GRID_STEP, duration=GRID_DURATION, text=text))
    return Transcript(language="ru", entries=entries)


class FFmpegSource(VideoSource):
    """Downloads by generating a deterministic local test video."""

    def __init__(self) -> None:
        self.info = VideoInfo(
            video_id="itg001",
            title="Интеграционная тестовая лекция",
            duration=VIDEO_DURATION,
            url="https://example.com/watch?v=itg001",
        )
        self.download_calls = 0

    def fetch_info(self, url: str) -> VideoInfo:
        return self.info

    def download(self, url: str, destination: Path) -> Path:
        self.download_calls += 1
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x120:rate=5:duration={VIDEO_DURATION}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={VIDEO_DURATION}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(destination),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr
        return destination


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        target_duration=50.0,
        duration_tolerance=10.0,
        workspace_dir=tmp_path / "workspace",
        blocks=BlockSettings(max_block_chars=400, max_block_seconds=200.0),
        selection=SelectionSettings(candidate_pool_factor=4.0),
    )


@pytest.fixture
def pipeline(tmp_path: Path) -> Pipeline:
    return Pipeline(
        settings=_settings(tmp_path),
        source=FFmpegSource(),
        transcripts=FakeTranscriptProvider(_grid_transcript()),
        llm=FakeLLM(target_duration=50.0, segment_duration=9.5),
    )


class TestEndToEnd:
    def test_full_pipeline_produces_trailer(self, pipeline: Pipeline, tmp_path: Path):
        trailer = pipeline.run("https://example.com/watch?v=itg001")
        assert trailer.exists()
        assert trailer.stat().st_size > 0

        workspace = tmp_path / "workspace" / "video_itg001"
        # All intermediate artifacts are saved.
        for relative in (
            "video_info.json",
            "source/original.mp4",
            "transcript/raw.json",
            "transcript/prepared.json",
            "analysis/blocks.json",
            "analysis/block_analysis.json",
            "analysis/content_map.json",
            "analysis/candidates.json",
            "trailer/script.json",
            "trailer/script_validated.json",
            "trailer/validation_report.json",
            "logs/pipeline.log",
        ):
            assert (workspace / relative).exists(), relative

        # Validated script keeps its duration in the configured window.
        validated = json.loads(
            (workspace / "trailer" / "script_validated.json").read_text(encoding="utf-8")
        )
        total = sum(s["end"] - s["start"] for s in validated["segments"])
        assert 40.0 <= total <= 60.0

        # The result decodes fully without errors.
        decode = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(trailer), "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert decode.returncode == 0, decode.stderr
        assert not decode.stderr.strip(), decode.stderr

    def test_rerun_uses_cached_artifacts(self, pipeline: Pipeline):
        url = "https://example.com/watch?v=itg001"
        first = pipeline.run(url)
        source = cast(FFmpegSource, pipeline._source)
        llm = cast(FakeLLM, pipeline._llm)
        downloads_after_first = source.download_calls
        calls_after_first = len(llm.calls)

        second = pipeline.run(url)
        assert second == first
        assert source.download_calls == downloads_after_first
        assert len(llm.calls) == calls_after_first

    def test_rerun_from_plan_recomputes_downstream_only(self, pipeline: Pipeline):
        url = "https://example.com/watch?v=itg001"
        pipeline.run(url)
        source = cast(FFmpegSource, pipeline._source)
        llm = cast(FakeLLM, pipeline._llm)
        downloads = source.download_calls
        calls = len(llm.calls)

        pipeline.run(url, from_stage="plan")
        assert source.download_calls == downloads  # видео не скачивается заново
        assert len(llm.calls) > calls  # сценарий пересчитан

    def test_rerun_from_download_forces_redownload(self, pipeline: Pipeline):
        url = "https://example.com/watch?v=itg001"
        pipeline.run(url)
        source = cast(FFmpegSource, pipeline._source)
        downloads = source.download_calls
        pipeline.run(url, from_stage="download")
        assert cast(FFmpegSource, pipeline._source).download_calls == downloads + 1
