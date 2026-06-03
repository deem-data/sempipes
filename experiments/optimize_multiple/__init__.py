from dataclasses import dataclass

from sempipes import LLM
from sempipes.optimisers.search_policy import SearchPolicy
from experiments.optimize_multiple.test_pipeline import TestPipeline


@dataclass(frozen=True)
class Setup:
    search: SearchPolicy
    num_trials: int
    cv: int
    llm_for_code_generation: LLM
    operator_selection_strategy: str = "ucb"
    optimize_all_operators: bool = False


__all__ = ["Setup", "TestPipeline"]
