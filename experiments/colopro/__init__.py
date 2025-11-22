from dataclasses import dataclass

from sempipes import LLM
from sempipes.optimisers.search_policy import SearchPolicy
from experiments.colopro.test_pipeline import TestPipeline


@dataclass(frozen=True)
class Setup:
    search: SearchPolicy
    num_trials: int
    llm_for_code_generation: LLM


__all__ = ["Setup", "TestPipeline"]
