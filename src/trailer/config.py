"""Centralised pipeline configuration (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    """Settings of the LLM provider (OpenAI-compatible or GigaChat)."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: SecretStr | None = None
    base_url: str | None = None
    scope: str | None = None  # GigaChat: GIGACHAT_API_PERS / _B2B / _CORP
    # GigaChat's chain is not trusted by default; set true or provide a CA bundle
    # (TRAILER_LLM__CA_BUNDLE_FILE) if the Russian Trusted CA is installed.
    verify_ssl_certs: bool = False
    ca_bundle_file: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    request_timeout: float = 120.0
    max_retries: int = 3


class BlockSettings(BaseModel):
    """Parameters of transcript segmentation into semantic blocks."""

    max_block_chars: int = 3500
    max_block_seconds: float = 300.0
    max_pause_inside_block: float = 4.0


class DownloadSettings(BaseModel):
    """Parameters of the source video download."""

    format: str = "bv*[height<=1080]+ba/b[height<=1080]/b"
    max_filesize_gb: float = 4.0


class SelectionSettings(BaseModel):
    """Parameters of candidate scoring and script validation."""

    candidate_pool_factor: float = 4.0
    min_candidate_duration: float = 5.0
    max_candidate_duration: float = 40.0
    dedupe_similarity: float = 0.6
    segment_padding: float = 0.4


class TracingSettings(BaseSettings):
    """Langfuse tracing settings; read from standard ``LANGFUSE_*`` variables."""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    public_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = None
    host: str | None = None  # deprecated Langfuse alias
    tracing_enabled: bool = True
    environment: str | None = None
    debug: bool = False


class Settings(BaseSettings):
    """Top-level pipeline settings; overridable via TRAILER_* env vars and .env."""

    model_config = SettingsConfigDict(
        env_prefix="TRAILER_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str | None = None
    language: str = "ru"
    target_duration: float = 90.0
    duration_tolerance: float = 10.0
    workspace_dir: Path = Path("workspace")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    blocks: BlockSettings = Field(default_factory=BlockSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)
    selection: SelectionSettings = Field(default_factory=SelectionSettings)

    @property
    def duration_min(self) -> float:
        return self.target_duration - self.duration_tolerance

    @property
    def duration_max(self) -> float:
        return self.target_duration + self.duration_tolerance
