from __future__ import annotations

import json
import re
from pathlib import Path

from trailer.analysis.llm.base import LLMClient
from trailer.domain.models import (
    BlockAnalysis,
    ContentMap,
    SegmentCandidate,
    Transcript,
    TranscriptEntry,
    TrailerScript,
    VideoInfo,
)
from trailer.sources.base import VideoSource
from trailer.transcription.base import TranscriptProvider

TOPICS = [
    "масштабирование распределённых систем",
    "проблемы традиционной монолитной архитектуры",
    "микросервисный подход к проектированию",
    "балансировка нагрузки и отказоустойчивость",
    "итоги и выводы по архитектуре",
]


def make_raw_transcript(
    total_seconds: float = 1800.0,
    cue_duration: float = 2.5,
    words_per_cue: int = 6,
) -> Transcript:
    """Deterministic Russian transcript with real-looking sentences."""
    entries: list[TranscriptEntry] = []
    t = 0.0
    sentence = 0
    while t < total_seconds:
        topic = TOPICS[sentence % len(TOPICS)]
        text = f"Итак, рассмотрим {topic} на конкретном примере из практики {sentence + 1}. " * 1
        text = text.strip()
        # Split into cues of about `words_per_cue` words.
        words = text.split()
        for i in range(0, len(words), words_per_cue):
            chunk = " ".join(words[i : i + words_per_cue])
            entries.append(
                TranscriptEntry(
                    start=t,
                    duration=cue_duration,
                    text=chunk,
                )
            )
            t += cue_duration
        sentence += 1
    return Transcript(language="ru", entries=entries)


class FakeSource(VideoSource):
    def __init__(self, duration: float = 1800.0) -> None:
        self.info = VideoInfo(
            video_id="fake123",
            title="Фейковая лекция о масштабировании",
            duration=duration,
            url="https://www.youtube.com/watch?v=fake123",
        )
        self.download_calls = 0

    def fetch_info(self, url: str) -> VideoInfo:
        return self.info

    def download(self, url: str, destination: Path) -> Path:
        self.download_calls += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake video bytes")
        return destination


class FakeTranscriptProvider(TranscriptProvider):
    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript
        self.fetch_calls = 0

    def fetch(self, video_id: str, language: str) -> Transcript:
        self.fetch_calls += 1
        return self._transcript


class FakeLLM(LLMClient):
    """Deterministic LLM that answers based on the requested schema."""

    def __init__(self, target_duration: float = 50.0, segment_duration: float = 10.0) -> None:
        self.calls: list[str] = []
        self.target_duration = target_duration
        self.segment_duration = segment_duration

    def complete_json(self, system_prompt: str, user_prompt: str, schema: type) -> object:
        self.calls.append(system_prompt.splitlines()[0])
        if schema is BlockAnalysis:
            return self._block_analysis(user_prompt)
        if schema is ContentMap:
            return ContentMap(
                main_topic="Масштабирование распределённых систем",
                narrative_flow=["проблема", "решение", "примеры", "выводы"],
                key_ideas=["Традиционный подход перестаёт работать при росте нагрузки."],
            )
        if schema is TrailerScript:
            return self._trailer_script(user_prompt)
        raise AssertionError(f"Unexpected schema: {schema}")

    def _block_analysis(self, user_prompt: str) -> BlockAnalysis:
        block_id = re.search(r"Идентификатор блока: (\S+)", user_prompt)
        bounds = re.search(r"Границы блока: от ([\d.]+) до ([\d.]+)", user_prompt)
        assert block_id and bounds, user_prompt[:200]
        block_name = block_id.group(1)
        start, end = float(bounds.group(1)), float(bounds.group(2))
        seg_start = start + 10.0
        seg_end = min(seg_start + self.segment_duration, end - 1.0)
        if seg_end - seg_start < 5.0:
            seg_start = start + 5.0
            seg_end = min(seg_start + self.segment_duration, end - 1.0)
        block_number = re.search(r"(\d+)", block_name)
        topic_index = int(block_number.group(1)) if block_number else 0
        return BlockAnalysis(
            block_id=block_name,
            summary="Автор объясняет проблему масштабирования.",
            topics=["масштабирование", "архитектура"],
            key_ideas=["Традиционный подход перестаёт работать при росте нагрузки."],
            candidates=[
                SegmentCandidate(
                    start=seg_start,
                    end=seg_end,
                    main_idea=f"{TOPICS[topic_index % len(TOPICS)]} — {block_name}",
                    context="Объяснение проблемы и конкретный пример.",
                    value=9,
                    self_contained=True,
                )
            ],
        )

    def _trailer_script(self, user_prompt: str) -> TrailerScript:
        candidates = [
            SegmentCandidate.model_validate(item)
            for item in _extract_json_array(user_prompt, "Проверенные кандидаты")
        ]
        segments = []
        total = 0.0
        for candidate in candidates:
            if total >= self.target_duration:
                break
            segments.append(
                {
                    "order": len(segments) + 1,
                    "start": candidate.start,
                    "end": candidate.end,
                    "purpose": "key_idea",
                    "main_idea": candidate.main_idea,
                    "reason": "Высокая смысловая ценность",
                }
            )
            total += candidate.end - candidate.start
        return TrailerScript.model_validate(
            {"target_duration": self.target_duration, "segments": segments}
        )


def _extract_json_array(user_prompt: str, header: str) -> list[dict]:
    part = user_prompt.split(header, 1)[1]
    start = part.index("[")
    end = part.index("\n\n", start)
    return json.loads(part[start:end].strip())
