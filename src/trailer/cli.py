"""Command-line interface of the trailer pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trailer.analysis.llm.provider import create_llm_client
from trailer.application.pipeline import STAGES, Pipeline
from trailer.config import Settings
from trailer.domain.exceptions import PipelineError, PipelineStageError
from trailer.observability.tracing import flush_tracing
from trailer.sources.youtube import YouTubeSource
from trailer.transcription.youtube_subtitles import YouTubeSubtitlesProvider


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailer.cli",
        description="Сборка содержательного трейлера из длинного видео по ссылке YouTube.",
    )
    parser.add_argument("--url", help="URL YouTube-видео (или задайте TRAILER_URL)")
    parser.add_argument("--duration", type=float, help="Целевая длительность трейлера, сек")
    parser.add_argument("--tolerance", type=float, help="Допустимое отклонение длительности, сек")
    parser.add_argument("--language", help="Язык субтитров (по умолчанию ru)")
    parser.add_argument(
        "--workspace", type=Path, help="Рабочая директория (по умолчанию workspace)"
    )
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        help="Пересчитать этот этап и все последующие, используя сохранённые данные",
    )
    parser.add_argument(
        "--only-stage",
        choices=STAGES,
        help="Выполнить только указанный этап (остальное загрузить из кеша)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересчитать все этапы, игнорируя сохранённые артефакты",
    )
    parser.add_argument("--verbose", action="store_true", help="Подробное логирование в консоль")
    return parser


def _setup_logging(verbose: bool) -> None:
    root = logging.getLogger("trailer")
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.url:
        overrides["url"] = args.url
    if args.duration:
        overrides["target_duration"] = args.duration
    if args.tolerance:
        overrides["duration_tolerance"] = args.tolerance
    if args.language:
        overrides["language"] = args.language
    if args.workspace:
        overrides["workspace_dir"] = args.workspace
    settings = Settings(**overrides)  # type: ignore[arg-type]
    _setup_logging(args.verbose)
    logger = logging.getLogger("trailer.cli")

    if not settings.url:
        parser.error("Не задан URL видео: используйте --url или переменную TRAILER_URL")

    try:
        pipeline = Pipeline(
            settings=settings,
            source=YouTubeSource(settings.download),
            transcripts=YouTubeSubtitlesProvider(),
            llm=create_llm_client(settings.llm),
        )
        trailer_path = pipeline.run(
            settings.url,
            from_stage=args.from_stage,
            only_stage=args.only_stage,
            force=args.force,
        )
    except PipelineStageError as exc:
        logger.error("Ошибка на этапе '%s': %s", exc.stage, exc)
        return 1
    except PipelineError as exc:
        logger.error("Ошибка конвейера: %s", exc)
        return 1
    finally:
        flush_tracing()

    logger.info("Готово. Трейлер: %s", trailer_path)
    print(trailer_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
