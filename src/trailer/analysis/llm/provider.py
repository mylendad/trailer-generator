from __future__ import annotations

import json
import logging
from typing import TypeVar

from openai import APIError, OpenAI
from pydantic import BaseModel, ValidationError

from trailer.config import LLMSettings
from trailer.domain.exceptions import LLMProviderError
from trailer.analysis.llm.base import LLMClient
from trailer.observability.tracing import record_usage, trace_generation

logger = logging.getLogger(__name__)

_STAGE = "llm"

T = TypeVar("T", bound=BaseModel)

_JSON_INSTRUCTIONS = "Отвечай ТОЛЬКО корректным JSON-объектом без markdown-обёртки и пояснений."


class OpenAICompatibleClient(LLMClient):
    """LLM client for any OpenAI-compatible chat completions API."""

    def __init__(self, settings: LLMSettings) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value():
            raise LLMProviderError(_STAGE, "Не задан API-ключ LLM (TRAILER_LLM__API_KEY)")
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            max_retries=1,  # retries are managed below to also cover validation
        )

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{_JSON_INSTRUCTIONS}"},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(1, self._settings.max_retries + 1):
            parameters = {
                "temperature": self._settings.temperature,
                "max_tokens": self._settings.max_tokens,
                "response_format": {"type": "json_object"},
            }
            with trace_generation(
                "llm.chat.completions",
                model=self._settings.model,
                input=messages,
                model_parameters=parameters,
                metadata={"provider": "openai", "attempt": attempt, "schema": schema.__name__},
            ) as generation:
                try:
                    response = self._client.chat.completions.create(
                        model=self._settings.model,
                        messages=messages,  # type: ignore[arg-type]
                        **parameters,
                    )
                    content = response.choices[0].message.content or ""
                    generation.update(output=content)
                    record_usage(generation, getattr(response, "usage", None))
                    data = json.loads(content)
                    return schema.model_validate(data)
                except APIError as exc:
                    last_error = exc
                    generation.update(level="ERROR", status_message=str(exc)[:1000])
                    logger.warning(
                        "Попытка %d/%d: ошибка LLM API: %s",
                        attempt,
                        self._settings.max_retries,
                        exc,
                    )
                except json.JSONDecodeError as exc:
                    last_error = exc
                    generation.update(level="ERROR", status_message=f"Некорректный JSON: {exc}")
                    logger.warning(
                        "Попытка %d/%d: некорректный JSON в ответе",
                        attempt,
                        self._settings.max_retries,
                    )
                except ValidationError as exc:
                    last_error = exc
                    generation.update(
                        level="ERROR", status_message=f"Ошибка схемы: {str(exc)[:500]}"
                    )
                    logger.warning(
                        "Попытка %d/%d: ответ не прошёл валидацию схемы: %s",
                        attempt,
                        self._settings.max_retries,
                        str(exc)[:500],
                    )
        raise LLMProviderError(
            _STAGE,
            f"LLM не вернула корректный ответ после {self._settings.max_retries} попыток: {last_error}",
        )


def create_llm_client(settings: LLMSettings) -> LLMClient:
    """Factory keeping provider selection in one place."""
    if settings.provider == "openai":
        return OpenAICompatibleClient(settings)
    if settings.provider == "gigachat":
        from trailer.analysis.llm.gigachat import GigachatClient

        return GigachatClient(settings)
    raise LLMProviderError(_STAGE, f"Неизвестный LLM-провайдер: {settings.provider!r}")
