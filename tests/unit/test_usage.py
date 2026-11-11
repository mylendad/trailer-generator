from __future__ import annotations

import trailer.observability.tracing as tracing
from trailer.observability.usage import TokenTotals, UsageTracker


def test_tracker_sums_usage():
    tracker = UsageTracker()
    tracker.add(10, 5)
    tracker.add(None, 3)
    tracker.add(None, None)
    totals = tracker.totals()
    assert totals == TokenTotals(prompt_tokens=10, completion_tokens=8)
    assert totals.total_tokens == 18


def test_tracker_reset():
    tracker = UsageTracker()
    tracker.add(7, 2)
    tracker.reset()
    assert tracker.totals() == TokenTotals(0, 0)


def test_token_totals_summary_and_dict():
    totals = TokenTotals(prompt_tokens=10, completion_tokens=4)
    assert totals.summary() == "входящие 10, исходящие 4, всего 14"
    assert totals.as_dict() == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }


def test_record_usage_feeds_tracker(monkeypatch):
    class _Dummy:
        def update(self, **kwargs: object) -> None:
            pass

    tracker = UsageTracker()
    monkeypatch.setattr(tracing, "TRACKER", tracker)
    tracing.record_usage(_Dummy(), {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
    tracing.record_usage(_Dummy(), {"input_tokens": 1, "output_tokens": 1})
    tracing.record_usage(_Dummy(), None)
    assert tracker.totals() == TokenTotals(prompt_tokens=4, completion_tokens=3)
