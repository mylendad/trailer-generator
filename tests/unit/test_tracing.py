from __future__ import annotations

from types import SimpleNamespace

import trailer.observability.tracing as tracing


class _Observation:
    def __init__(self) -> None:
        self.updates: dict[str, object] = {}

    def update(self, **kwargs: object) -> None:
        self.updates.update(kwargs)


class _DisabledSettings:
    tracing_enabled = True
    public_key = None
    secret_key = None
    base_url = None
    host = None
    environment = None
    debug = False


def _reset(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "TracingSettings", _DisabledSettings)
    tracing._initialized = False
    tracing._client = None


def test_span_and_generation_are_noops_without_keys(monkeypatch):
    _reset(monkeypatch)
    with tracing.trace_span("stage", input={"a": 1}) as span:
        span.update(output="ok")
    with tracing.trace_generation("llm", model="m", input="prompt") as generation:
        tracing.record_usage(
            generation, SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        )
    assert isinstance(span, tracing._NoopObservation)
    tracing.flush_tracing()
    tracing._initialized = False
    tracing._client = None


def test_record_usage_from_object():
    observation = _Observation()
    tracing.record_usage(
        observation, SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert observation.updates["usage_details"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_record_usage_from_dict_sums_missing_total():
    observation = _Observation()
    tracing.record_usage(observation, {"input_tokens": 7, "output_tokens": 2})
    assert observation.updates["usage_details"] == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }


def test_record_usage_ignores_missing_usage():
    observation = _Observation()
    tracing.record_usage(observation, None)
    assert observation.updates == {}


def test_enabled_tracing_creates_spans_and_flushes(monkeypatch):
    import sys
    from types import SimpleNamespace

    class FakeObservation:
        def __init__(self):
            self.kwargs = {}
            self.updates = []

        def update(self, **kwargs):
            self.updates.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeClient:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.observations = []
            self.flushed = False
            FakeClient.instances.append(self)

        def start_as_current_observation(self, **kwargs):
            observation = FakeObservation()
            observation.kwargs = kwargs
            self.observations.append(observation)
            return observation

        def flush(self):
            self.flushed = True

    class EnabledSettings(_DisabledSettings):
        public_key = "pk"
        secret_key = "sk"
        base_url = "http://langfuse.test"
        environment = "test"

    monkeypatch.setattr(tracing, "TracingSettings", EnabledSettings)
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=FakeClient))
    tracing._initialized = False
    tracing._client = None
    try:
        with tracing.trace_span("pipeline.plan", input={"x": 1}) as span:
            span.update(output="done")
        with tracing.trace_generation("llm.chat", model="m", input="prompt") as generation:
            tracing.record_usage(generation, {"prompt_tokens": 1, "completion_tokens": 2})
        tracing.flush_tracing()
    finally:
        tracing._initialized = False
        tracing._client = None

    client = FakeClient.instances[-1]
    assert client.kwargs["base_url"] == "http://langfuse.test"
    assert client.kwargs["environment"] == "test"
    assert client.flushed
    assert [obs.kwargs["as_type"] for obs in client.observations] == ["span", "generation"]
    assert client.observations[1].updates[-1]["usage_details"]["total_tokens"] == 3
