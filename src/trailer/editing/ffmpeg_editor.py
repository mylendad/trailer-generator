from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from trailer.domain.exceptions import EditingError
from trailer.domain.models import TrailerScript
from trailer.editing.ffmpeg_binary import resolve_ffmpeg

logger = logging.getLogger(__name__)

_STAGE = "render"

_STDERR_TAIL = 2000

_ENCODE_FLAGS = (
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-ar",
    "44100",
    "-ac",
    "2",
)

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RESOLUTION_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(\d+(?:\.\d+)?)")


def build_cut_command(
    source: Path,
    output: Path,
    start: float,
    end: float,
    executable: str = "ffmpeg",
) -> list[str]:
    """Re-encoding cut command; guarantees exact boundaries and uniform segments."""
    return [
        executable,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output),
    ]


def build_concat_command(
    list_file: Path,
    output: Path,
    executable: str = "ffmpeg",
) -> list[str]:
    return [
        executable,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output),
    ]


def build_normalize_command(
    intro: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
    executable: str = "ffmpeg",
) -> list[str]:
    """Re-encode the intro to match the source resolution, fps and audio."""
    return [
        executable,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(intro),
        "-vf",
        f"scale={width}:{height},fps={fps:g},setsar=1,format=yuv420p",
        *_ENCODE_FLAGS,
        str(output),
    ]


def build_crossfade_command(
    clips: list[tuple[Path, float]],
    transition: float,
    output: Path,
    executable: str = "ffmpeg",
) -> list[str]:
    """Chain clips with video crossfades and matching audio crossfades.

    ``clips`` is an ordered list of ``(path, duration)`` pairs; durations are
    probed from the actual files, because xfade offsets must be exact.
    """
    if len(clips) < 2:
        raise EditingError(_STAGE, "Для crossfade нужно минимум два фрагмента")
    args: list[str] = [executable, "-y", "-loglevel", "error"]
    for path, _ in clips:
        args += ["-i", str(path)]
    filters: list[str] = []
    offset = clips[0][1]
    prev_video, prev_audio = "0:v", "0:a"
    for i, (_, duration) in enumerate(clips[1:], start=1):
        offset -= transition
        video_out, audio_out = f"v{i}", f"a{i}"
        filters.append(
            f"[{prev_video}][{i}:v]xfade=transition=fade:duration={transition:.3f}"
            f":offset={offset:.3f}[{video_out}]"
        )
        filters.append(f"[{prev_audio}][{i}:a]acrossfade=d={transition:.3f}[{audio_out}]")
        prev_video, prev_audio = video_out, audio_out
        offset += duration
    args += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        f"[{prev_video}]",
        "-map",
        f"[{prev_audio}]",
        *_ENCODE_FLAGS,
        str(output),
    ]
    return args


class FFmpegEditor:
    """Cuts source intervals, adds the intro and assembles the final MP4.

    The editor makes no content decisions: it blindly renders the validated
    script it receives. It only fixes one mechanical defect: segment ends are
    pulled forward to the next real speech pause so words are never clipped.
    """

    def __init__(
        self,
        source: Path,
        *,
        intro_path: Path | None = None,
        outro_path: Path | None = None,
        transition_duration: float = 0.5,
        max_speech_extension: float = 10.0,
    ) -> None:
        self._source = source
        self._intro_path = intro_path
        self._outro_path = outro_path
        self._transition = transition_duration
        self._max_speech_extension = max_speech_extension
        ffmpeg = resolve_ffmpeg()
        if ffmpeg is None:
            raise EditingError(
                _STAGE,
                "FFmpeg не найден. Установите ffmpeg в PATH или пакет imageio-ffmpeg.",
            )
        self._ffmpeg = ffmpeg
        self._durations: dict[Path, float] = {}

    def build_trailer(self, script: TrailerScript, output: Path, workdir: Path) -> Path:
        workdir.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)

        silences = self._detect_silences(self._source)

        segment_files: list[Path] = []
        for segment in sorted(script.segments, key=lambda s: s.order):
            start = self._speech_start_at(segment.start, silences)
            end = self._speech_pause_after(segment.end, silences)
            if start != segment.start:
                logger.info(
                    "Фрагмент %d: начало сдвинуто к паузе речи %.2f -> %.2f",
                    segment.order,
                    segment.start,
                    start,
                )
            if end != segment.end:
                logger.info(
                    "Фрагмент %d: конец сдвинут к паузе речи %.2f -> %.2f",
                    segment.order,
                    segment.end,
                    end,
                )
            segment_path = (workdir / f"segment_{segment.order:02d}.mp4").resolve()
            command = build_cut_command(
                self._source,
                segment_path,
                start,
                end,
                executable=self._ffmpeg,
            )
            self._run(command, f"нарезка фрагмента {segment.order}")
            segment_files.append(segment_path)

        intro_file = self._prepare_media(self._intro_path, workdir, "intro.mp4", "интро")
        outro_file = self._prepare_media(self._outro_path, workdir, "outro.mp4", "аутро")
        if self._transition <= 0:
            self._concat(intro_file, segment_files, outro_file, output, workdir)
        else:
            self._crossfade(intro_file, segment_files, outro_file, output)

        if not output.exists() or output.stat().st_size == 0:
            raise EditingError(_STAGE, f"Итоговый файл не создан или пуст: {output}")
        extras = [label for label, file in (("интро", intro_file), ("аутро", outro_file)) if file]
        logger.info(
            "Трейлер собран: %s (%.1f МБ, %d фрагментов%s)",
            output,
            output.stat().st_size / 1e6,
            len(segment_files),
            f", с {' и '.join(extras)}" if extras else "",
        )
        return output

    # ---------------------------------------------------------------- helpers

    def _prepare_media(
        self, source: Path | None, workdir: Path, target_name: str, label: str
    ) -> Path | None:
        if source is None:
            return None
        width, height, fps = self._probe_stream(self._source)
        normalized = workdir / target_name
        self._run(
            build_normalize_command(
                source, normalized, width, height, fps, executable=self._ffmpeg
            ),
            f"нормализация {label}",
        )
        return normalized

    def _concat(
        self,
        intro_file: Path | None,
        segment_files: list[Path],
        outro_file: Path | None,
        output: Path,
        workdir: Path,
    ) -> None:
        entries = [intro_file] if intro_file else []
        entries += segment_files
        if outro_file:
            entries.append(outro_file)
        list_file = workdir / "concat.txt"
        list_file.write_text(
            "\n".join(f"file '{path}'" for path in entries) + "\n",
            encoding="utf-8",
        )
        self._run(
            build_concat_command(list_file, output, executable=self._ffmpeg),
            "склейка трейлера",
        )

    def _crossfade(
        self,
        intro_file: Path | None,
        segment_files: list[Path],
        outro_file: Path | None,
        output: Path,
    ) -> None:
        clips: list[tuple[Path, float]] = []
        if intro_file is not None:
            clips.append((intro_file, self._probe_duration(intro_file)))
        for path in segment_files:
            clips.append((path, self._probe_duration(path)))
        if outro_file is not None:
            clips.append((outro_file, self._probe_duration(outro_file)))
        self._run(
            build_crossfade_command(clips, self._transition, output, executable=self._ffmpeg),
            "склейка трейлера с переходами",
        )

    def _probe_duration(self, path: Path) -> float:
        cached = self._durations.get(path)
        if cached is not None:
            return cached
        info = self._probe(path, "определение длительности")
        match = _DURATION_RE.search(info)
        if not match:
            raise EditingError(_STAGE, f"Не удалось определить длительность: {path}")
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        self._durations[path] = duration
        return duration

    def _probe_stream(self, path: Path) -> tuple[int, int, float]:
        info = self._probe(path, "определение параметров видео")
        resolution = _RESOLUTION_RE.search(info)
        fps = _FPS_RE.search(info)
        if not resolution or not fps:
            raise EditingError(_STAGE, f"Не удалось определить параметры видео: {path}")
        return int(resolution.group(1)), int(resolution.group(2)), float(fps.group(1))

    def _probe(self, path: Path, action: str) -> str:
        try:
            completed = subprocess.run(
                [self._ffmpeg, "-i", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise EditingError(_STAGE, f"Не удалось запустить FFmpeg ({action}): {exc}") from exc
        return completed.stderr

    def _detect_silences(self, path: Path) -> list[tuple[float, float]]:
        """Real speech pauses of the source: (start, end) pairs of silences."""
        try:
            completed = subprocess.run(
                [
                    self._ffmpeg,
                    "-i",
                    str(path),
                    "-af",
                    "silencedetect=noise=-25dB:d=0.2",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise EditingError(
                _STAGE, f"Не удалось запустить FFmpeg (детект тишины): {exc}"
            ) from exc
        stderr = completed.stderr
        starts = [float(m) for m in _SILENCE_START_RE.findall(stderr)]
        ends = [float(m) for m in _SILENCE_END_RE.findall(stderr)]
        if not starts:
            return []
        if len(ends) < len(starts):  # trailing silence has no silence_end line
            ends.append(starts[-1] + 1e9)
        return list(zip(starts, ends))

    def _speech_pause_after(self, end: float, silences: list[tuple[float, float]]) -> float:
        """Move the cut to the start of the first real silence after ``end``.

        If ``end`` already sits inside a silence (or no silence follows within
        the allowed extension), the boundary is kept as is.
        """
        for silence_start, silence_end in silences:
            if silence_start <= end <= silence_end:
                return end
        for silence_start, _silence_end in silences:
            if silence_start >= end - 0.05:
                if silence_start > end + self._max_speech_extension:
                    return end
                return max(end, silence_start)
        return end

    def _speech_start_at(self, start: float, silences: list[tuple[float, float]]) -> float:
        """Drop a mid-speech fragment tail before ``start`` up to a speech pause.

        A candidate that starts in the middle of the previous thought gets its
        beginning moved forward to the moment speech resumes after the first
        silence. If ``start`` already sits inside a silence, it is kept.
        """
        for silence_start, silence_end in silences:
            if silence_start <= start <= silence_end:
                return start
        for _silence_start, silence_end in silences:
            if silence_end >= start - 0.05:
                if silence_end > start + self._max_speech_extension:
                    return start
                return max(start, silence_end)
        return start

    def _run(self, command: list[str], action: str) -> None:
        logger.debug("Запуск: %s", " ".join(command))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise EditingError(_STAGE, f"Не удалось запустить FFmpeg ({action}): {exc}") from exc
        if completed.returncode != 0:
            stderr_tail = completed.stderr[-_STDERR_TAIL:]
            raise EditingError(_STAGE, f"FFmpeg ошибся при выполнении ({action}): {stderr_tail}")
