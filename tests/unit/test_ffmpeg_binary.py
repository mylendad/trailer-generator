from __future__ import annotations

from pathlib import Path

import pytest

import trailer.editing.ffmpeg_binary as ffmpeg_binary


def test_prefers_system_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_binary.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    assert ffmpeg_binary.resolve_ffmpeg() == "/usr/bin/ffmpeg"


def test_falls_back_to_bundled_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundled = tmp_path / "ffmpeg-linux-x86_64"
    bundled.write_bytes(b"bin")
    import imageio_ffmpeg

    monkeypatch.setattr(ffmpeg_binary.shutil, "which", lambda _: None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(bundled))
    assert ffmpeg_binary.resolve_ffmpeg() == str(bundled)


def test_returns_none_when_bundled_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import imageio_ffmpeg

    monkeypatch.setattr(ffmpeg_binary.shutil, "which", lambda _: None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/nonexistent/ffmpeg")
    assert ffmpeg_binary.resolve_ffmpeg() is None
