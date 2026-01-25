from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class LLM:
    name: str
    parameters: dict = field(default_factory=dict)


@dataclass_json
@dataclass(frozen=True)
class Config:
    llm_for_code_generation: LLM = LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 0.0},
    )
    llm_for_batch_processing: LLM = LLM(name="gemini/gemini-2.5-flash", parameters={"temperature": 0.0})
    batch_size_for_batch_processing: int = 20


# Holds the current config; ContextVar makes it safe for threads/async tasks
_CONFIG: ContextVar[Config] = ContextVar("_CONFIG", default=Config())


def get_config() -> Config:
    """Read-only access to the current configuration."""
    return _CONFIG.get()


def _set_config(cfg: Config) -> None:
    """Set the process-wide default config (for the current thread/async task)."""
    _CONFIG.set(cfg)


def ensure_default_config() -> None:
    """Ensure that the default config is set (useful for tests)."""
    _set_config(Config())


def update_config(**kwargs) -> None:
    """Update specific attributes of the current config by creating a new Config instance."""
    current_config = get_config()

    # Create a new config with updated attributes
    updated_config = Config(
        llm_for_code_generation=kwargs.get("llm_for_code_generation", current_config.llm_for_code_generation),
        llm_for_batch_processing=kwargs.get("llm_for_batch_processing", current_config.llm_for_batch_processing),
        batch_size_for_batch_processing=kwargs.get(
            "batch_size_for_batch_processing", current_config.batch_size_for_batch_processing
        ),
    )

    _set_config(updated_config)
