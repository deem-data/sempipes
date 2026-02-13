import copy
import math
import random
from typing import Any

from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name

from sempipes.logging import get_logger
from sempipes.optimisers.search_policy import _ROOT_TRIAL, Outcome, SearchNode, SearchPolicy

logger = get_logger()


class EvolutionarySearch(SearchPolicy):
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.root_node: SearchNode | None = None
        self.outcomes: list[Outcome] = []

    def clone_empty(self) -> SearchPolicy:
        return EvolutionarySearch(population_size=self.population_size)

    def create_root_node(self, dag_sink: DataOp, operator_name: str):
        data_op = find_node_by_name(dag_sink, operator_name)
        print(data_op)
        assert data_op is not None
        assert data_op._skrub_impl is not None
        assert data_op._skrub_impl.estimator is not None

        empty_state = data_op._skrub_impl.estimator.empty_state()

        logger.info("EVO_SEARCH> Creating root node")
        root_node = SearchNode(
            trial=_ROOT_TRIAL,
            parent_trial=None,
            memory=[],
            predefined_state=empty_state,
            parent_score=None,
        )

        self.root_node = root_node
        return root_node

    def record_outcome(
        self,
        search_node: SearchNode,
        operator_state: dict[str, Any],
        score: float,
        operator_memory_update: str,
    ):
        outcome = Outcome(
            state=operator_state,  # type: ignore[arg-type]
            score=score,
            search_node=search_node,
            memory_update=operator_memory_update,
        )
        self.outcomes.append(outcome)

    def create_next_search_node(self) -> SearchNode | None:
        assert self.root_node is not None

        best_population = sorted(self.outcomes, key=lambda x: x.score, reverse=True)[: self.population_size]
        best_population = [outcome for outcome in best_population if math.isfinite(outcome.score)]

        if len(best_population) == 0:
            raise ValueError("EVO_SEARCH> No finite scores found in the best population!!! This should not happen.")

        total_score = sum(outcome.score for outcome in best_population)
        probabilities = [outcome.score / total_score for outcome in best_population]
        outcome_to_evolve = random.choices(best_population, weights=probabilities, k=1)[0]

        logger.info(f"EVO_SEARCH> Trying to improve node with score {outcome_to_evolve.score}")
        updated_memory = copy.deepcopy(outcome_to_evolve.search_node.memory)
        updated_memory.append({"update": outcome_to_evolve.memory_update, "score": outcome_to_evolve.score})
        next_node = SearchNode(
            trial=None,
            parent_trial=outcome_to_evolve.search_node.trial,
            memory=updated_memory,
            predefined_state=None,
            parent_score=outcome_to_evolve.score,
        )

        return next_node

    def get_outcomes(self):
        return self.outcomes
