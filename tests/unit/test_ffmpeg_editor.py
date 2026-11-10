from __future__ import annotations

from pathlib import Path

import pytest

from trailer.domain.exceptions import EditingError
from trailer.domain.models import TrailerScript
from trailer.editing.ffmpeg_editor import FFmpegEditor, build_concat_command, build_cut_command


class TestCommandBuilders:
    def test_cut_command_structure(self, tmp_path: Path):
        command = build_cut_command(Path("in.mp4"), tmp_path / "seg.mp4", 12.5, 30.0)
        assert command[0] == "ffmpeg"
        assert command[command.index("-ss") + 1] == "12.500"
        assert command[command.index("-to") + 1] == "30.000"
        assert command[command.index("-i") + 1] == "in.mp4"
        assert "libx264" in command
        assert "aac" in command
        assert command[-1].endswith("seg.mp4")

    def test_concat_command_structure(self, tmp_path: Path):
        command = build_concat_command(tmp_path / "list.txt", tmp_path / "out.mp4")
        assert "-f" in command and "concat" in command
        assert command[command.index("-c") + 1] == "copy"


class TestFFmpegEditor:
    def _editor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FFmpegEditor:
        monkeypatch.setattr(
            "trailer.editing.ffmpeg_editor.resolve_ffmpeg", lambda: "/usr/bin/ffmpeg"
        )
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake")
        return FFmpegEditor(source)

    def _script(self) -> TrailerScript:
        return TrailerScript.model_validate(
            {
                "target_duration": 20.0,
                "segments": [
                    {"order": 1, "start": 1.0, "end": 6.0, "purpose": "hook", "main_idea": "a"},
                    {
                        "order": 2,
                        "start": 20.0,
                        "end": 26.0,
                        "purpose": "key_idea",
                        "main_idea": "b",
                    },
                ],
            }
        )

    def test_missing_ffmpeg_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("trailer.editing.ffmpeg_editor.resolve_ffmpeg", lambda: None)
        from trailer.editing.ffmpeg_editor import FFmpegEditor as Editor

        with pytest.raises(EditingError, match="FFmpeg не найден"):
            Editor(tmp_path / "src.mp4")

    def test_build_trailer_runs_commands_and_creates_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        editor = self._editor(tmp_path, monkeypatch)
        calls: list[list[str]] = []

        def fake_run(command, **kwargs):
            calls.append(command)
            # Emulate ffmpeg creating the output file.
            output = Path(command[-1])
            if command[1] != "-y":
                output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"segment-data")

            class Result:
                returncode = 0
                stderr = ""

            return Result()

        monkeypatch.setattr("trailer.editing.ffmpeg_editor.subprocess.run", fake_run)
        output = editor.build_trailer(self._script(), tmp_path / "trailer.mp4", tmp_path / "segs")
        assert output.exists()
        assert len(calls) == 3  # 2 cuts + 1 concat
        assert calls[-1][calls[-1].index("-f") + 1] == "concat"
        concat_list = (tmp_path / "segs" / "concat.txt").read_text(encoding="utf-8")
        assert all(line.startswith("file '/") for line in concat_list.splitlines() if line)

    def test_ffmpeg_failure_raises_with_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        editor = self._editor(tmp_path, monkeypatch)

        def fake_run(command, **kwargs):
            class Result:
                returncode = 1
                stderr = "boom"

            return Result()

        monkeypatch.setattr("trailer.editing.ffmpeg_editor.subprocess.run", fake_run)
        with pytest.raises(EditingError, match="boom"):
            editor.build_trailer(self._script(), tmp_path / "trailer.mp4", tmp_path / "segs")
