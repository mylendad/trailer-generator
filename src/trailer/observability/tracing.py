"""Optional Langfuse tracing.

Tracing is a no-op unless ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``
are set (in the environment or ``.env``). The Langfuse SDK is imported lazily
so the pipeline keeps working when tracing is not configured.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from trailer.config import TracingSettings
from trailer.observability.usage import TRACKER

logger = logging.getLogger(__name__)

_client: Any | None = None
_initialized = False


class _NoopObservation:
    """Stand-in that accepts the same ``update`` calls as a Langfuse span."""

    def update(self, **_: Any) -> None:
        return None


def _get_client() -> Any | None:
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    try:
        settings = TracingSettings()
    except Exception as exc:  # invalid .env should never break the pipeline
        logger.warning("Langfuse: не удалось прочитать настройки: %s", exc)
        return None
    if not settings.tracing_enabled:
        logger.info("Langfuse: трассировка отключена (LANGFUSE_TRACING_ENABLED=false)")
        return None
    if not settings.public_key or not settings.secret_key:
        logger.info("Langfuse: ключи не заданы, трассировка отключена")
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("Langfuse: пакет langfuse не установлен, трассировка отключена")
        return None
    kwargs: dict[str, Any] = {
        "public_key": settings.public_key,
        "secret_key": settings.secret_key,
        "debug": settings.debug,
    }
    base_url = settings.base_url or settings.host
    if base_url:
        kwargs["base_url"] = base_url
    if settings.environment:
        kwargs["environment"] = settings.environment
    try:
        _client = Langfuse(**kwargs)
    except Exception as exc:
        logger.warning("Langfuse: не удалось инициализировать клиент: %s", exc)
        return None
    logger.info("Langfuse: трассировка включена (%s)", base_url or "cloud.langfuse.com")
    return _client


@contextmanager
def trace_span(
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Create a span; nested observations automatically become children."""
    client = _get_client()
    if client is None:
        yield _NoopObservation()
        return
    with client.start_as_current_observation(as_type="span", name=name) as observation:
        try:
            observation.update(input=input, metadata=metadata)
            yield observation
        except Exception as exc:
            observation.update(level="ERROR", status_message=str(exc)[:1000])
            raise


@contextmanager
def trace_generation(
    name: str,
    *,
    model: str | None,
    input: Any = None,
    model_parameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Create a generation observation around a single model call."""
    client = _get_client()
    if client is None:
        yield _NoopObservation()
        return
    with client.start_as_current_observation(
        as_type="generation", name=name, model=model
    ) as observation:
        try:
            observation.update(input=input, model_parameters=model_parameters, metadata=metadata)
            yield observation
        except Exception as exc:
            observation.update(level="ERROR", status_message=str(exc)[:1000])
            raise


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def record_usage(observation: Any, usage: Any | None) -> None:
    """Record token counts from OpenAI/GigaChat usage objects."""
    if usage is None:
        return
    prompt = _usage_value(usage, "prompt_tokens", "input_tokens")
    completion = _usage_value(usage, "completion_tokens", "output_tokens")
    total = _usage_value(usage, "total_tokens")
    details: dict[str, int] = {}
    if prompt is not None:
        details["prompt_tokens"] = prompt
    if completion is not None:
        details["completion_tokens"] = completion
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if total is not None:
        details["total_tokens"] = total
    if details:
        observation.update(usage_details=details)
    TRACKER.add(prompt, completion)


def flush_tracing() -> None:
    """Flush pending spans; safe to call when tracing is disabled."""
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("Langfuse: не удалось отправить трейсы: %s", exc)
