import math
import random

from sempipes.logging import get_logger
from sempipes.optimisers.search_policy import SearchNode, SearchPolicy

logger = get_logger()


class EvolutionarySearch(SearchPolicy):
    def __init__(self, population_size: int):
        super().__init__()
        self.population_size = population_size

    def clone_empty(self) -> SearchPolicy:
        return EvolutionarySearch(population_size=self.population_size)

    def create_next_search_node(self, trial: int, operator_to_evolve: str, all_operator_names: list[str]) -> SearchNode:
        assert self.root_node is not None

        best_population = sorted(self.outcomes, key=lambda x: x.score, reverse=True)[: self.population_size]
        best_population = [outcome for outcome in best_population if math.isfinite(outcome.score)]

        if len(best_population) == 0:
            raise ValueError("EVO_SEARCH> No finite scores found in the best population!!! This should not happen.")

        total_score = sum(outcome.score for outcome in best_population)
        probabilities = [outcome.score / total_score for outcome in best_population]
        outcome_to_evolve = random.choices(best_population, weights=probabilities, k=1)[0]

        return self._expand_tree(trial, operator_to_evolve, all_operator_names, outcome_to_evolve)

        # fixed_operators = [name for name in all_operator_names if name != operator_to_evolve]
        # fixed_operator_states = OperatorStates()
        # for operator_name in fixed_operators:
        #     assert outcome_to_evolve.states.exists_for(operator_name)
        #     fixed_operator_states.set(operator_name, outcome_to_evolve.states.get(operator_name))

        # if outcome_to_evolve.memory_update is None:
        #     memory_update = OptimisableMixin.EMPTY_MEMORY_UPDATE
        # else:
        #     memory_update = outcome_to_evolve.memory_update

        # logger.info(f"EVO_SEARCH> Trying to improve node with score {outcome_to_evolve.score}")
        # updated_memories = copy.deepcopy(outcome_to_evolve.search_node.memories)
        # updated_memories.append(operator_to_evolve, {"update": memory_update, "score": outcome_to_evolve.score})

        # next_node = SearchNode(
        #     trial=trial,
        #     parent_trial=outcome_to_evolve.search_node.trial,
        #     operator_to_evolve=operator_to_evolve,
        #     memories=updated_memories,
        #     fixed_states=fixed_operator_states,
        #     parent_score=outcome_to_evolve.score,
        # )

        # return next_node
