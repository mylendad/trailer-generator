from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Provider-agnostic interface for structured LLM calls."""

    @abstractmethod
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        """Ask the model and return its answer validated against ``schema``."""
