"""Domain exceptions of the trailer pipeline."""


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class PipelineStageError(PipelineError):
    """An error bound to a concrete pipeline stage."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class InvalidURLError(PipelineStageError):
    def __init__(self, url: str) -> None:
        super().__init__("fetch_info", f"Некорректная ссылка на видео: {url!r}")


class SourceError(PipelineStageError):
    """Failure while fetching video metadata or downloading the file."""


class TranscriptionError(PipelineStageError):
    """Failure while fetching subtitles."""


class LLMProviderError(PipelineStageError):
    """Failure of the LLM API or of its structured response."""


class ScriptValidationError(PipelineStageError):
    """The trailer script failed validation against source data."""


class ValidationError(PipelineError):
    """Generic data validation failure (not tied to a stage)."""


class EditingError(PipelineStageError):
    """Failure inside the FFmpeg-based editor."""
