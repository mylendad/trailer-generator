from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import trailer.sources.youtube as youtube
from trailer.config import DownloadSettings


class _FakeYDL:
    last_options: dict[str, Any] = {}

    def __init__(self, options: dict[str, Any]) -> None:
        _FakeYDL.last_options = options

    def __enter__(self) -> "_FakeYDL":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def download(self, urls: list[str]) -> None:
        template = _FakeYDL.last_options["outtmpl"]
        Path(template.replace("%(ext)s", "mkv")).write_bytes(b"video")


def _download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ffmpeg: str | None) -> dict:
    monkeypatch.setattr(youtube, "resolve_ffmpeg", lambda: ffmpeg)
    monkeypatch.setattr(youtube.yt_dlp, "YoutubeDL", _FakeYDL)
    source = youtube.YouTubeSource(DownloadSettings())
    source.download("https://youtu.be/x", tmp_path / "original.mp4")
    return _FakeYDL.last_options


def test_download_renames_chosen_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options = _download(tmp_path, monkeypatch, "/usr/bin/ffmpeg")
    assert options["outtmpl"].endswith(".%(ext)s")
    assert (tmp_path / "original.mp4").read_bytes() == b"video"
    assert not (tmp_path / "original.mkv").exists()


def test_download_passes_ffmpeg_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options = _download(tmp_path, monkeypatch, "/usr/bin/ffmpeg")
    assert options["ffmpeg_location"] == "/usr/bin/ffmpeg"
    assert options["format"] == DownloadSettings().format


def test_download_without_ffmpeg_avoids_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = _download(tmp_path, monkeypatch, None)
    assert "ffmpeg_location" not in options
    assert "+" not in options["format"]
