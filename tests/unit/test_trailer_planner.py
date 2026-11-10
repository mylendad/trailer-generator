from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from trailer.analysis.trailer_planner import TrailerPlanner
from trailer.config import Settings
from trailer.domain.models import ContentMap, SegmentCandidate, TrailerScript


class _FakeLLM:
    def __init__(self, script: TrailerScript) -> None:
        self._script = script

    def complete_json(
        self, system_prompt: str, user_prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        return self._script


def _script(*durations: float) -> TrailerScript:
    start = 0.0
    segments = []
    for order, duration in enumerate(durations, start=1):
        segments.append(
            {
                "order": order,
                "start": start,
                "end": start + duration,
                "purpose": "hook" if order == 1 else "key_idea",
                "main_idea": f"идея {order}",
            }
        )
        start += duration
    return TrailerScript.model_validate({"target_duration": 90.0, "segments": segments})


def _planner(tmp_path: Path) -> TrailerPlanner:
    settings = Settings(target_duration=90.0, duration_tolerance=10.0, workspace_dir=tmp_path)
    return TrailerPlanner(_FakeLLM(_script(40.0, 40.0, 40.0)), settings)  # type: ignore[arg-type]


def test_trims_overlong_script_preserving_hook_and_conclusion(tmp_path: Path) -> None:
    planner = _planner(tmp_path)
    candidates = [SegmentCandidate(start=0, end=40, main_idea="a", value=7)]

    result = planner.build_script(ContentMap(main_topic="тема"), candidates)

    assert len(result.segments) == 2
    assert result.segments[0].purpose == "hook"
    assert [s.order for s in result.segments] == [1, 2]
    assert result.total_duration <= 100.0


def test_keeps_script_that_already_fits(tmp_path: Path) -> None:
    settings = Settings(target_duration=90.0, duration_tolerance=10.0, workspace_dir=tmp_path)
    planner = TrailerPlanner(_FakeLLM(_script(30.0, 30.0)), settings)  # type: ignore[arg-type]

    result = planner.build_script(
        ContentMap(main_topic="тема"),
        [SegmentCandidate(start=0, end=30, main_idea="a", value=7)],
    )

    assert len(result.segments) == 2
    assert result.total_duration == 60.0
