from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    llm_for_code_generation: str = "openai/gpt-4.1"
    llm_settings_for_code_generation: dict = field(default_factory=lambda: {"temperature": 0.0})
    llm_for_batch_processing: str = "ollama/gpt-oss:20b"
    llm_settings_for_batch_processing: dict = field(
        default_factory=lambda: {"api_base": "http://localhost:11434", "temperature": 0.0}
    )


# Holds the current config; ContextVar makes it safe for threads/async tasks
_CONFIG: ContextVar[Config] = ContextVar("_CONFIG", default=Config())


def get_config() -> Config:
    """Read-only access to the current configuration."""
    return _CONFIG.get()


def set_config(cfg: Config) -> None:
    """Set the process-wide default config (for the current thread/async task)."""
    _CONFIG.set(cfg)


@contextmanager
def override_config(**overrides):
    """
    Temporarily override selected fields for a block/scope.
    Works with nested overrides and is task/thread-safe.
    """
    current = get_config()
    new_cfg = Config(**{**current.__dict__, **overrides})
    token = _CONFIG.set(new_cfg)
    try:
        yield new_cfg
    finally:
        _CONFIG.reset(token)
