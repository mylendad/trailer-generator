from __future__ import annotations

from pathlib import Path

import pytest

from trailer.domain.exceptions import EditingError
from trailer.domain.models import TrailerScript
from trailer.editing.ffmpeg_editor import (
    FFmpegEditor,
    build_concat_command,
    build_crossfade_command,
    build_cut_command,
    build_normalize_command,
)


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

    def test_normalize_command_structure(self, tmp_path: Path):
        command = build_normalize_command(
            tmp_path / "intro.mp4", tmp_path / "norm.mp4", 1920, 1080, 24.0
        )
        assert command[command.index("-i") + 1].endswith("intro.mp4")
        assert command[command.index("-vf") + 1] == "scale=1920:1080,fps=24,setsar=1,format=yuv420p"
        assert "libx264" in command and "aac" in command

    def test_crossfade_command_structure(self, tmp_path: Path):
        clips = [
            (Path("intro.mp4"), 13.01),
            (Path("seg1.mp4"), 33.48),
            (Path("seg2.mp4"), 29.06),
        ]
        command = build_crossfade_command(clips, 0.5, tmp_path / "out.mp4")
        assert command[command.index("-i") + 1] == "intro.mp4"
        filters = command[command.index("-filter_complex") + 1]
        assert "xfade=transition=fade:duration=0.500:offset=12.510" in filters
        assert "offset=45.490" in filters
        assert filters.count("acrossfade=d=0.500") == 2
        assert command[command.index("-map") + 1] == "[v2]"
        assert command[command.index("-map", command.index("-map") + 1) + 1] == "[a2]"

    def test_crossfade_requires_two_clips(self, tmp_path: Path):
        with pytest.raises(EditingError, match="минимум два"):
            build_crossfade_command([(Path("a.mp4"), 5.0)], 0.5, tmp_path / "out.mp4")


class TestFFmpegEditor:
    def _editor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        **kwargs,
    ) -> FFmpegEditor:
        monkeypatch.setattr(
            "trailer.editing.ffmpeg_editor.resolve_ffmpeg", lambda: "/usr/bin/ffmpeg"
        )
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake")
        return FFmpegEditor(source, **kwargs)

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
        editor = self._editor(tmp_path, monkeypatch, transition_duration=0.0)
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
        assert len(calls) == 4  # detect-silences + 2 cuts + 1 concat
        assert calls[1][calls[1].index("-ss") + 1] == "1.000"
        assert calls[-1][calls[-1].index("-f") + 1] == "concat"
        concat_list = (tmp_path / "segs" / "concat.txt").read_text(encoding="utf-8")
        assert all(line.startswith("file '/") for line in concat_list.splitlines() if line)

    def test_build_trailer_with_intro_and_crossfade(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        intro = tmp_path / "intro_raw.mp4"
        intro.write_bytes(b"fake-intro")
        editor = self._editor(tmp_path, monkeypatch, intro_path=intro, transition_duration=0.5)
        calls: list[list[str]] = []

        def fake_run(command, **kwargs):
            calls.append(command)
            Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[-1]).write_bytes(b"data")

            class Result:
                returncode = 0
                stderr = (
                    "Duration: 00:00:13.01, start: 0.000000\n"
                    "Stream #0:0: Video: h264, 1920x1080, 24 fps\n"
                )

            return Result()

        monkeypatch.setattr("trailer.editing.ffmpeg_editor.subprocess.run", fake_run)
        output = editor.build_trailer(self._script(), tmp_path / "trailer.mp4", tmp_path / "segs")
        assert output.exists()
        # detect-silences + 2 cuts + probe(source) + normalize + 3 probes + crossfade
        assert len(calls) == 9
        assert calls[0][calls[0].index("-af") + 1].startswith("silencedetect")
        normalize = next(c for c in calls if "-vf" in c)
        assert normalize[normalize.index("-vf") + 1].startswith("scale=1920:1080")
        final = calls[-1]
        assert "-filter_complex" in final
        filters = final[final.index("-filter_complex") + 1]
        assert "offset=12.510" in filters

    def test_build_trailer_with_outro_crossfade(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        outro = tmp_path / "outro_raw.mp4"
        outro.write_bytes(b"fake-outro")
        editor = self._editor(tmp_path, monkeypatch, outro_path=outro, transition_duration=0.5)
        calls: list[list[str]] = []

        def fake_run(command, **kwargs):
            calls.append(command)
            Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[-1]).write_bytes(b"data")

            class Result:
                returncode = 0
                stderr = (
                    "Duration: 00:00:08.00, start: 0.000000\n"
                    "Stream #0:0: Video: h264, 1920x1080, 24 fps\n"
                )

            return Result()

        monkeypatch.setattr("trailer.editing.ffmpeg_editor.subprocess.run", fake_run)
        output = editor.build_trailer(self._script(), tmp_path / "trailer.mp4", tmp_path / "segs")
        assert output.exists()
        # detect + 2 cuts + probe(source) + normalize(outro) + probes + crossfade
        normalize = next(c for c in calls if "-vf" in c)
        assert normalize[normalize.index("-i") + 1].endswith("outro_raw.mp4")
        final = calls[-1]
        filters = final[final.index("-filter_complex") + 1]
        # outro is the last input in the chain
        assert final[final.index("-i") + 1].endswith("segs/segment_01.mp4")
        assert "offset=" in filters

    def test_probe_duration_parses_stderr(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        editor = self._editor(tmp_path, monkeypatch)

        def fake_run(command, **kwargs):
            class Result:
                returncode = 1
                stderr = "Input #0 ...\n  Duration: 00:01:05.50, start: 0.000000\n"

            return Result()

        monkeypatch.setattr("trailer.editing.ffmpeg_editor.subprocess.run", fake_run)
        assert editor._probe_duration(Path("x.mp4")) == 65.5

    def test_detect_silences_parses_stderr(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        editor = self._editor(tmp_path, monkeypatch)

        def fake_run(command, **kwargs):
            class Result:
                returncode = 0
                stderr = (
                    "[silencedetect @ 0x0] silence_start: 3442.95\n"
                    "[silencedetect @ 0x0] silence_end: 3447.5 | silence_duration: 4.55\n"
                    "[silencedetect @ 0x0] silence_start: 3500.1\n"
                )

            return Result()

        monkeypatch.setattr("trailer.editing.ffmpeg_editor.subprocess.run", fake_run)
        assert editor._detect_silences(Path("x.mp4")) == [(3442.95, 3447.5), (3500.1, 3500.1 + 1e9)]

    def test_speech_pause_after_extends_to_next_silence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        editor = self._editor(tmp_path, monkeypatch)
        silences = [(3442.95, 3447.5), (3455.0, 3456.0)]
        assert editor._speech_pause_after(3436.28, silences) == pytest.approx(3442.95)
        assert editor._speech_pause_after(3450.0, silences) == pytest.approx(3455.0)

    def test_speech_pause_after_inside_silence_or_beyond_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        editor = self._editor(tmp_path, monkeypatch)
        silences = [(3442.95, 3447.5)]
        assert editor._speech_pause_after(3445.0, silences) == pytest.approx(3445.0)
        assert editor._speech_pause_after(3440.0, silences) == pytest.approx(3442.95)
        # No silence within the extension cap -> keep the boundary.
        assert editor._speech_pause_after(3999.0, silences) == pytest.approx(3999.0)

    def test_speech_start_at_trims_tail_to_pause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        editor = self._editor(tmp_path, monkeypatch)
        silences = [(3430.0, 3430.5), (3440.0, 3442.0)]
        # start sits inside a silence -> kept
        assert editor._speech_start_at(3441.0, silences) == pytest.approx(3441.0)
        # start is mid-speech: the tail before the pause is dropped, speech
        # resumes at the silence end
        assert editor._speech_start_at(3435.5, silences) == pytest.approx(3442.0)
        # the first resume point is beyond the cap -> keep the boundary
        assert editor._speech_start_at(3431.5, silences) == pytest.approx(3431.5)
        # no silence after start at all -> keep
        assert editor._speech_start_at(3500.0, silences) == pytest.approx(3500.0)

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
