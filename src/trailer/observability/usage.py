"""Process-wide LLM token accounting."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenTotals:
    """Aggregated token counts for one pipeline run."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        return (
            f"входящие {self.prompt_tokens}, исходящие {self.completion_tokens}, "
            f"всего {self.total_tokens}"
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class UsageTracker:
    """Thread-safe accumulator of LLM usage across all calls in the process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def add(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        with self._lock:
            self._prompt_tokens += prompt_tokens or 0
            self._completion_tokens += completion_tokens or 0

    def totals(self) -> TokenTotals:
        with self._lock:
            return TokenTotals(self._prompt_tokens, self._completion_tokens)

    def reset(self) -> None:
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0


TRACKER = UsageTracker()
