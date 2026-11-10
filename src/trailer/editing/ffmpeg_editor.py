from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from trailer.domain.exceptions import EditingError
from trailer.domain.models import TrailerScript
from trailer.editing.ffmpeg_binary import resolve_ffmpeg

logger = logging.getLogger(__name__)

_STAGE = "render"

_STDERR_TAIL = 2000


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


class FFmpegEditor:
    """Cuts source intervals and concatenates them into the final MP4.

    The editor makes no content decisions: it blindly renders the validated
    script it receives.
    """

    def __init__(self, source: Path) -> None:
        self._source = source
        ffmpeg = resolve_ffmpeg()
        if ffmpeg is None:
            raise EditingError(
                _STAGE,
                "FFmpeg не найден. Установите ffmpeg в PATH или пакет imageio-ffmpeg.",
            )
        self._ffmpeg = ffmpeg

    def build_trailer(self, script: TrailerScript, output: Path, workdir: Path) -> Path:
        workdir.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)

        segment_files: list[Path] = []
        for segment in sorted(script.segments, key=lambda s: s.order):
            segment_path = (workdir / f"segment_{segment.order:02d}.mp4").resolve()
            command = build_cut_command(
                self._source,
                segment_path,
                segment.start,
                segment.end,
                executable=self._ffmpeg,
            )
            self._run(command, f"нарезка фрагмента {segment.order}")
            segment_files.append(segment_path)

        list_file = workdir / "concat.txt"
        list_file.write_text(
            "\n".join(f"file '{path}'" for path in segment_files) + "\n",
            encoding="utf-8",
        )
        self._run(
            build_concat_command(list_file, output, executable=self._ffmpeg),
            "склейка трейлера",
        )

        if not output.exists() or output.stat().st_size == 0:
            raise EditingError(_STAGE, f"Итоговый файл не создан или пуст: {output}")
        logger.info(
            "Трейлер собран: %s (%.1f МБ, %d фрагментов)",
            output,
            output.stat().st_size / 1e6,
            len(segment_files),
        )
        return output

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
