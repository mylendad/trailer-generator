from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, SecretStr

from trailer.analysis.llm import GigachatClient, create_llm_client
from trailer.config import LLMSettings
from trailer.domain.exceptions import LLMProviderError


class _Answer(BaseModel):
    value: str


def _settings(**overrides: object) -> LLMSettings:
    base: dict[str, object] = {
        "provider": "gigachat",
        "model": "GigaChat-2-Max",
        "api_key": SecretStr("secret"),
        "max_retries": 2,
    }
    base.update(overrides)
    return LLMSettings(**base)  # type: ignore[arg-type]


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(LLMProviderError):
        create_llm_client(LLMSettings(provider="unknown", api_key=SecretStr("x")))


def test_gigachat_requires_api_key() -> None:
    with pytest.raises(LLMProviderError):
        GigachatClient(_settings(api_key=None))


def test_factory_creates_gigachat(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeGiga:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("trailer.analysis.llm.gigachat.GigaChat", FakeGiga)
    client = create_llm_client(_settings())

    assert isinstance(client, GigachatClient)
    assert captured["credentials"] == "secret"
    assert captured["model"] == "GigaChat-2-Max"
    assert "scope" not in captured


def test_gigachat_accepts_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeGiga:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("trailer.analysis.llm.gigachat.GigaChat", FakeGiga)
    token = "header.payload.signature.part.five"

    GigachatClient(_settings(api_key=SecretStr(token)))

    assert captured["access_token"] == token
    assert "credentials" not in captured


def test_gigachat_replaces_generic_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeGiga:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("trailer.analysis.llm.gigachat.GigaChat", FakeGiga)

    GigachatClient(_settings(model="gigachat"))

    assert captured["model"] == "GigaChat-2-Max"


def test_gigachat_uses_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGiga:
        def __init__(self, **kwargs: object) -> None:
            pass

        def chat_parse(
            self,
            payload: dict[str, object],
            *,
            response_format: type[BaseModel],
            strict: bool = True,
        ) -> tuple[None, BaseModel]:
            return None, response_format.model_validate({"value": "structured"})

        def chat(self, payload: dict[str, object]) -> object:
            raise AssertionError("plain chat must not be used")

    monkeypatch.setattr("trailer.analysis.llm.gigachat.GigaChat", FakeGiga)

    assert GigachatClient(_settings()).complete_json("sys", "user", _Answer) == _Answer(
        value="structured"
    )


def test_gigachat_falls_back_to_plain_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGiga:
        def __init__(self, **kwargs: object) -> None:
            pass

        def chat_parse(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("structured output unavailable")

        def chat(self, payload: dict[str, object]) -> object:
            message = SimpleNamespace(content='```json\n{"value": "fallback"}\n```')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr("trailer.analysis.llm.gigachat.GigaChat", FakeGiga)

    assert GigachatClient(_settings()).complete_json("sys", "user", _Answer) == _Answer(
        value="fallback"
    )


def test_gigachat_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    class FakeGiga:
        def __init__(self, **kwargs: object) -> None:
            pass

        def chat_parse(self, *args: object, **kwargs: object) -> object:
            calls["count"] += 1
            raise RuntimeError("boom")

        def chat(self, payload: dict[str, object]) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr("trailer.analysis.llm.gigachat.GigaChat", FakeGiga)

    with pytest.raises(LLMProviderError):
        GigachatClient(_settings(max_retries=2)).complete_json("sys", "user", _Answer)

    assert calls["count"] == 2
