from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from gigachat import GigaChat
from pydantic import BaseModel, ValidationError

from trailer.analysis.llm.base import LLMClient
from trailer.config import LLMSettings
from trailer.domain.exceptions import LLMProviderError
from trailer.observability.tracing import record_usage, trace_generation

logger = logging.getLogger(__name__)

_STAGE = "llm"

T = TypeVar("T", bound=BaseModel)

_JSON_INSTRUCTIONS = "Отвечай ТОЛЬКО корректным JSON-объектом без markdown-обёртки и пояснений."

_DEFAULT_MODEL = "GigaChat-2-Max"

_GENERIC_MODELS = {"", "gigachat", "default"}


def _extract_json(text: str) -> Any:
    """Extract the first JSON object from a possibly chatty model answer."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("В ответе LLM нет JSON-объекта")
    return json.loads(text[start : end + 1])


def _auth_kwargs(api_key: str) -> dict[str, str]:
    """Authorization key (base64 ``id:secret``) or an already issued JWE token."""
    if api_key.count(".") >= 2 or len(api_key) > 400:
        return {"access_token": api_key}
    return {"credentials": api_key}


class GigachatClient(LLMClient):
    """LLM client for Sber GigaChat (structured output via JSON schema)."""

    def __init__(self, settings: LLMSettings) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value():
            raise LLMProviderError(
                _STAGE,
                "Не задан ключ авторизации GigaChat (TRAILER_LLM__API_KEY)",
            )
        self._settings = settings
        kwargs: dict[str, Any] = {
            "timeout": settings.request_timeout,
            "max_retries": 1,  # retries are managed below to also cover validation
            "verify_ssl_certs": settings.verify_ssl_certs,
            "ca_bundle_file": settings.ca_bundle_file,
            **_auth_kwargs(settings.api_key.get_secret_value()),
        }
        model = (settings.model or "").strip()
        if model.lower() in _GENERIC_MODELS:
            model = _DEFAULT_MODEL
        kwargs["model"] = model
        self._model_name = model
        if settings.scope:
            kwargs["scope"] = settings.scope
        self._client = GigaChat(**kwargs)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        payload = {
            "messages": [
                {"role": "system", "content": f"{system_prompt}\n\n{_JSON_INSTRUCTIONS}"},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        parameters = {
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(1, self._settings.max_retries + 1):
            with trace_generation(
                "llm.gigachat.chat_parse",
                model=self._model_name,
                input=payload["messages"],
                model_parameters={**parameters, "response_format": schema.__name__},
                metadata={"provider": "gigachat", "attempt": attempt, "schema": schema.__name__},
            ) as generation:
                try:
                    completion, parsed = self._client.chat_parse(payload, response_format=schema)
                    if completion is not None:
                        generation.update(output=completion.choices[0].message.content)
                        record_usage(generation, getattr(completion, "usage", None))
                    return parsed
                except Exception as exc:  # chat_parse is beta and may be unsupported
                    generation.update(level="ERROR", status_message=str(exc)[:1000])
                    logger.warning(
                        "Попытка %d/%d: structured output GigaChat недоступен: %s",
                        attempt,
                        self._settings.max_retries,
                        exc,
                    )
            with trace_generation(
                "llm.gigachat.chat",
                model=self._model_name,
                input=payload["messages"],
                model_parameters=parameters,
                metadata={"provider": "gigachat", "attempt": attempt, "schema": schema.__name__},
            ) as generation:
                try:
                    completion = self._client.chat(payload)
                    content = completion.choices[0].message.content or ""
                    generation.update(output=content)
                    record_usage(generation, getattr(completion, "usage", None))
                    return schema.model_validate(_extract_json(content))
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
                except (ValueError, json.JSONDecodeError, IndexError) as exc:
                    last_error = exc
                    generation.update(level="ERROR", status_message=f"Некорректный JSON: {exc}")
                    logger.warning(
                        "Попытка %d/%d: некорректный JSON в ответе",
                        attempt,
                        self._settings.max_retries,
                    )
                except Exception as exc:
                    last_error = exc
                    generation.update(level="ERROR", status_message=str(exc)[:1000])
                    logger.warning(
                        "Попытка %d/%d: ошибка GigaChat API: %s",
                        attempt,
                        self._settings.max_retries,
                        exc,
                    )
        message = (
            f"GigaChat не вернула корректный ответ после "
            f"{self._settings.max_retries} попыток: {last_error}"
        )
        if last_error is not None and "Token has expired" in str(last_error):
            message = (
                "Токен GigaChat истёк. Укажите в TRAILER_LLM__API_KEY Authorization key "
                "(base64 от client_id:client_secret) из личного кабинета GigaChat, "
                "а не access token."
            )
        raise LLMProviderError(_STAGE, message)
