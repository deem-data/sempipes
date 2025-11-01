from abc import ABC, abstractmethod
from dataclasses import dataclass

from sempipes import LLM
from sempipes.optimisers.search_policy import SearchPolicy


@dataclass(frozen=True)
class Setup:
    search: SearchPolicy
    sample_size: int
    num_trials: int
    llm_for_code_generation: LLM


class TestPipeline(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def baseline(self) -> float:
        pass

    @abstractmethod
    def optimize(self, setup: Setup) -> float:
        pass


__all__ = ["Setup", "TestPipeline"]
