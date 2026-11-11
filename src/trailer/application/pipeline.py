"""Pipeline orchestration: stages, caching of intermediate artifacts, logging."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from trailer.analysis.block_analyzer import BlockAnalyzer
from trailer.analysis.candidate_selector import CandidateSelector, SelectionResult
from trailer.analysis.content_mapper import ContentMapper
from trailer.analysis.llm.base import LLMClient
from trailer.analysis.segmenter import TranscriptSegmenter
from trailer.analysis.transcript_preprocessor import TranscriptPreprocessor
from trailer.analysis.trailer_planner import TrailerPlanner
from trailer.config import Settings
from trailer.domain.exceptions import PipelineStageError, SourceError
from trailer.domain.models import (
    BlockAnalysis,
    ContentMap,
    PreparedTranscript,
    SegmentCandidate,
    Transcript,
    TranscriptBlock,
    TrailerScript,
    VideoInfo,
)
from trailer.editing.ffmpeg_editor import FFmpegEditor
from trailer.observability.tracing import trace_span
from trailer.observability.usage import TRACKER
from trailer.sources.base import VideoSource
from trailer.transcription.base import TranscriptProvider
from trailer.validation.trailer_validator import TrailerValidator

from .workspace import Workspace

logger = logging.getLogger(__name__)

T = TypeVar("T")

STAGES = (
    "fetch_info",
    "download",
    "transcript",
    "prepare",
    "segment",
    "analyze",
    "map",
    "select",
    "plan",
    "validate",
    "render",
)


class Pipeline:
    """Runs the full pipeline, reusing artifacts saved by previous runs."""

    def __init__(
        self,
        settings: Settings,
        source: VideoSource,
        transcripts: TranscriptProvider,
        llm: LLMClient,
    ) -> None:
        self._settings = settings
        self._source = source
        self._transcripts = transcripts
        self._llm = llm
        self._rerun: set[str] = set()

    def run(
        self,
        url: str,
        from_stage: str | None = None,
        only_stage: str | None = None,
        force: bool = False,
    ) -> Path:
        """Run the pipeline and return the path to the resulting trailer.

        Stages whose artifacts already exist on disk are skipped unless they
        are in the re-run set (``from_stage``, ``only_stage``, ``force``).
        """
        self._rerun = self._compute_rerun(from_stage, only_stage, force)
        with trace_span(
            "trailer.pipeline",
            input={"url": url},
            metadata={
                "from_stage": from_stage,
                "only_stage": only_stage,
                "force": force,
                "target_duration": self._settings.target_duration,
                "language": self._settings.language,
            },
        ) as span:
            try:
                return self._run_pipeline(span, url)
            finally:
                totals = TRACKER.totals()
                logger.info("Токены LLM за прогон: %s", totals.summary())
                span.update(metadata={"token_totals": totals.as_dict()})

    def _run_pipeline(self, span: Any, url: str) -> Path:
        logger.info("Запуск конвейера: %s", url)
        if self._rerun:
            logger.info("Этапы к пересчёту: %s", ", ".join(sorted(self._rerun)))

        info = self._stage_fetch_info(url)
        workspace = Workspace(self._settings.workspace_dir / f"video_{info.video_id}")
        workspace.ensure_directories()
        self._attach_file_logging(workspace)

        self._stage_download(url, workspace, info)
        transcript = self._stage_model(
            "transcript",
            workspace.raw_transcript,
            Transcript,
            lambda: self._transcripts.fetch(info.video_id, self._settings.language),
        )
        prepared = self._stage_model(
            "prepare",
            workspace.prepared_transcript,
            PreparedTranscript,
            lambda: TranscriptPreprocessor().prepare(transcript),
        )
        blocks = self._stage(
            "segment",
            workspace.blocks,
            compute=lambda: TranscriptSegmenter(self._settings.blocks).segment(prepared),
            load=lambda path: [
                TranscriptBlock.model_validate(item) for item in _read_json(path)["blocks"]
            ],
            dump=lambda path, value: _save_json(
                path, {"blocks": [block.model_dump(mode="json") for block in value]}
            ),
        )
        analyses = self._stage(
            "analyze",
            workspace.block_analysis,
            compute=lambda: BlockAnalyzer(self._llm).analyze_all(blocks),
            load=lambda path: [
                BlockAnalysis.model_validate(item) for item in _read_json(path)["analyses"]
            ],
            dump=lambda path, value: _save_json(
                path, {"analyses": [analysis.model_dump(mode="json") for analysis in value]}
            ),
        )
        content_map = self._stage_model(
            "map",
            workspace.content_map,
            ContentMap,
            lambda: ContentMapper(self._llm).build_map(analyses),
        )
        selection = self._stage(
            "select",
            workspace.candidates,
            compute=lambda: CandidateSelector(
                self._settings.selection, self._settings.target_duration, transcript
            ).select(analyses),
            load=_load_selection,
            dump=_dump_selection,
        )
        script = self._stage_model(
            "plan",
            workspace.script,
            TrailerScript,
            lambda: TrailerPlanner(self._llm, self._settings).build_script(
                content_map, selection.accepted
            ),
        )
        validated = self._stage_validate(workspace, script, transcript, prepared, info)
        trailer = self._stage_render(workspace, validated)
        span.update(output={"trailer": str(trailer), "video_id": info.video_id})
        return trailer

    # ------------------------------------------------------------------ stages

    def _stage_fetch_info(self, url: str) -> VideoInfo:
        cache_root = self._settings.workspace_dir
        cached = self._find_cached_video_info(url, cache_root)
        if cached is not None and "fetch_info" not in self._rerun:
            logger.info("Метаданные видео взяты из кеша: %s", cached.video_id)
            return cached
        info = self._source.fetch_info(url)
        workspace = Workspace(cache_root / f"video_{info.video_id}")
        workspace.ensure_directories()
        _save_model(workspace.video_info, info)
        _append_index_entry(cache_root / "video_index.json", url, info.video_id)
        return info

    def _find_cached_video_info(self, url: str, cache_root: Path) -> VideoInfo | None:
        index_path = cache_root / "video_index.json"
        if not index_path.exists():
            return None
        try:
            index = _read_json(index_path)
            for entry in index:
                if entry.get("url") == url:
                    info_path = cache_root / entry["dir"] / "video_info.json"
                    return VideoInfo.model_validate(_read_json(info_path))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            logger.debug("Кеш метаданных недоступен, выполняем запрос")
        return None

    def _stage_download(self, url: str, workspace: Workspace, info: VideoInfo) -> Path:
        if (
            "download" not in self._rerun
            and workspace.original_video.exists()
            and workspace.original_video.stat().st_size > 0
        ):
            logger.info("Видео уже загружено: %s", workspace.original_video)
            return workspace.original_video
        if workspace.original_video.exists():
            workspace.original_video.unlink()
        self._source.download(url, workspace.original_video)
        _save_model(workspace.video_info, info)
        return workspace.original_video

    def _stage_validate(
        self,
        workspace: Workspace,
        script: TrailerScript,
        raw: Transcript,
        prepared: PreparedTranscript,
        info: VideoInfo,
    ) -> TrailerScript:
        if "validate" not in self._rerun and workspace.validated_script.exists():
            logger.info("Проверенный сценарий уже существует: %s", workspace.validated_script)
            return TrailerScript.model_validate(_read_json(workspace.validated_script))
        validator = TrailerValidator(self._settings, prepared, info.duration, raw)
        validated, report = validator.build_validated_script(script)
        _save_model(workspace.validated_script, validated)
        _save_json(
            workspace.validation_report,
            {
                "total_duration": report.total_duration,
                "kept": len(report.kept),
                "rejected": report.rejected,
                "adjustments": report.adjustments,
            },
        )
        return validated

    def _stage_render(self, workspace: Workspace, script: TrailerScript) -> Path:
        cached = (
            "render" not in self._rerun
            and workspace.trailer_video.exists()
            and workspace.trailer_video.stat().st_size > 0
        )
        with trace_span(
            "pipeline.render",
            input={
                "segments": [
                    {"order": seg.order, "start": seg.start, "end": seg.end}
                    for seg in script.segments
                ],
                "total_duration": sum(seg.duration for seg in script.segments),
                "intro": self._resolve_intro(),
                "transition_duration": self._settings.editing.transition_duration,
            },
            metadata={"cached": cached},
        ) as span:
            if cached:
                logger.info("Трейлер уже собран: %s", workspace.trailer_video)
                span.update(output={"path": str(workspace.trailer_video), "cached": True})
                return workspace.trailer_video
            if not workspace.original_video.exists():
                raise SourceError("render", "Исходное видео не найдено в рабочей директории")
            editor = FFmpegEditor(
                workspace.original_video,
                intro_path=self._resolve_intro(),
                transition_duration=self._settings.editing.transition_duration,
            )
            trailer = editor.build_trailer(
                script, workspace.trailer_video, workspace.trailer_segments_dir
            )
            span.update(
                output={
                    "path": str(trailer),
                    "cached": False,
                    "size_bytes": trailer.stat().st_size,
                }
            )
            return trailer

    # ------------------------------------------------------------------ helpers

    def _compute_rerun(
        self, from_stage: str | None, only_stage: str | None, force: bool
    ) -> set[str]:
        if from_stage is not None and only_stage is not None:
            raise ValueError("--from-stage и --only-stage взаимоисключающие")
        for stage in (from_stage, only_stage):
            if stage is not None and stage not in STAGES:
                raise ValueError(
                    f"Неизвестный этап {stage!r}. Доступные этапы: {', '.join(STAGES)}"
                )
        if force:
            return set(STAGES)
        if from_stage is not None:
            return set(STAGES[STAGES.index(from_stage) :])
        if only_stage is not None:
            return {only_stage}
        # Default: recompute nothing that already has artifacts on disk.
        return set()

    def _stage[T](
        self,
        stage: str,
        artifact: Path,
        compute: Callable[[], T],
        load: Callable[[Path], T],
        dump: Callable[[Path, T], None],
    ) -> T:
        cached = stage not in self._rerun and artifact.exists()
        with trace_span(
            f"pipeline.{stage}",
            input={"artifact": str(artifact)},
            metadata={"stage": stage, "cached": cached},
        ) as span:
            if cached:
                logger.info("Этап %s: используем сохранённый результат %s", stage, artifact)
                result = load(artifact)
                span.update(output={"cached": True, "artifact": str(artifact)})
                return result
            logger.info("Этап %s: выполняем", stage)
            try:
                result = compute()
            except PipelineStageError:
                raise
            except Exception as exc:
                raise PipelineStageError(stage, str(exc)) from exc
            dump(artifact, result)
            loaded = load(artifact)
            logger.info("Этап %s: завершён", stage)
            span.update(output={"cached": False, "artifact": str(artifact)})
            return loaded

    def _stage_model[M: BaseModel](
        self,
        stage: str,
        artifact: Path,
        model: type[M],
        compute: Callable[[], M],
    ) -> M:
        def load(path: Path) -> M:
            return model.model_validate(_read_json(path))

        return self._stage(stage, artifact, compute, load, _save_model)

    def _resolve_intro(self) -> Path | None:
        intro_dir = self._settings.editing.intro_dir
        if not intro_dir.is_dir():
            return None
        files = sorted(intro_dir.glob("*.mp4"))
        if not files:
            return None
        logger.info("Интро найдено: %s", files[0])
        return files[0]

    def _attach_file_logging(self, workspace: Workspace) -> None:
        root = logging.getLogger("trailer")
        for handler in list(root.handlers):
            is_file = isinstance(handler, logging.FileHandler)
            if is_file and Path(getattr(handler, "baseFilename", "")) == workspace.log_file:
                return
        handler = logging.FileHandler(workspace.log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.setLevel(logging.DEBUG)
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_model(path: Path, model: BaseModel) -> None:
    _save_json(path, model)


def _append_index_entry(index_path: Path, url: str, video_id: str) -> None:
    try:
        index: list = _read_json(index_path) if index_path.exists() else []
    except (OSError, json.JSONDecodeError):
        index = []
    index = [entry for entry in index if entry.get("url") != url]
    index.append({"url": url, "video_id": video_id, "dir": f"video_{video_id}"})
    _save_json(index_path, index)


def _dump_selection(path: Path, selection: SelectionResult) -> None:
    _save_json(
        path,
        {
            "accepted": [candidate.model_dump(mode="json") for candidate in selection.accepted],
            "rejected_count": selection.rejected_count,
            "reasons": selection.reasons,
        },
    )


def _load_selection(path: Path) -> SelectionResult:
    data = _read_json(path)
    return SelectionResult(
        accepted=[SegmentCandidate.model_validate(item) for item in data["accepted"]],
        rejected_count=data.get("rejected_count", 0),
        reasons=data.get("reasons", []),
    )
