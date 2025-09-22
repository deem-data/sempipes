from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLM:
    name: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    llm_for_code_generation: LLM = LLM(
        name="openai/gpt-4.1",
        parameters={"temperature": 0.0},
    )
    llm_for_batch_processing: LLM = LLM(
        name="ollama/gpt-oss:20b",
        parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
    )


# Holds the current config; ContextVar makes it safe for threads/async tasks
_CONFIG: ContextVar[Config] = ContextVar("_CONFIG", default=Config())


def get_config() -> Config:
    """Read-only access to the current configuration."""
    return _CONFIG.get()


def set_config(cfg: Config) -> None:
    """Set the process-wide default config (for the current thread/async task)."""
    _CONFIG.set(cfg)
