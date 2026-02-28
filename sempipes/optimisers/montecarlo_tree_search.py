from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skrub import DataOp

from sempipes.logging import get_logger
from sempipes.optimisers.search_policy import _ROOT_TRIAL, SearchNode, SearchPolicy

logger = get_logger()


@dataclass
class SearchNodeStatistics:
    scores: list[float] = field(default_factory=list)

    def visits(self) -> int:
        return len(self.scores)


class MonteCarloTreeSearch(SearchPolicy):
    def __init__(self, c: float = 1.41, root_children: int = 3, max_non_root_children: int = 2):
        super().__init__()
        self.c = c
        self.root_children = root_children
        self.max_non_root_children = max_non_root_children
        self.min_score = float("inf")
        self.max_score = -float("inf")
        self.search_node_stats: dict[int, SearchNodeStatistics] = {}

    def clone_empty(self) -> SearchPolicy:
        return MonteCarloTreeSearch(
            c=self.c,
            root_children=self.root_children,
            max_non_root_children=self.max_non_root_children,
        )

    def create_root_node(self, dag_sink: DataOp, all_operator_names: list[str]) -> SearchNode:
        self.search_node_stats[_ROOT_TRIAL] = SearchNodeStatistics()
        return super().create_root_node(dag_sink, all_operator_names)

    def record_outcome(
        self,
        search_node: SearchNode,
        operator_states: dict[str, Any] | None,
        score: float,
        operator_memory_updates: dict[str, str],
    ):
        super().record_outcome(search_node, operator_states, score, operator_memory_updates)

        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)

        assert search_node.trial is not None
        if search_node.trial not in self.search_node_stats:
            self.search_node_stats[search_node.trial] = SearchNodeStatistics()
        self.search_node_stats[search_node.trial].scores.append(score)
        self._back_propagate(search_node, score)

    def _back_propagate(self, leaf_node: SearchNode, score: float):
        current_node: SearchNode | None = leaf_node
        while current_node is not None:
            assert current_node.trial in self.search_node_stats
            node_stats = self.search_node_stats[current_node.trial]
            node_stats.scores.append(score)

            if current_node.parent_trial is not None:
                parent_outcome = next(
                    filter(lambda outcome: outcome.search_node.trial == current_node.parent_trial, self.outcomes),  # type: ignore[arg-type]
                    None,
                )
                assert parent_outcome is not None
                current_node = parent_outcome.search_node
            else:
                current_node = None

    def _uct(self, search_node: SearchNode) -> float:
        parent_node_outcome = next(
            filter(lambda outcome: outcome.search_node.trial == search_node.parent_trial, self.outcomes), None
        )
        assert parent_node_outcome is not None
        parent_node = parent_node_outcome.search_node
        assert search_node.trial is not None
        node_stats = self.search_node_stats[search_node.trial]
        assert parent_node.trial is not None
        parent_node_stats = self.search_node_stats[parent_node.trial]

        normalized_scores = [
            (score - self.min_score) / (self.max_score - self.min_score + 1e-8) for score in node_stats.scores
        ]

        w_i = np.sum(normalized_scores)
        n_i = node_stats.visits()
        N_i = parent_node_stats.visits()
        assert n_i > 0
        assert N_i > 0

        uct_value = (w_i / n_i) + self.c * np.sqrt(np.log(N_i) / n_i)
        # TODO Enable again through config
        # logger.info(
        #     f"UCT of node {search_node.trial}: w_i: {w_i}, n_i: {n_i}, N_i: {N_i} -> "
        #     f"{(w_i / n_i)} + {self.c * np.sqrt(np.log(N_i) / n_i)} = {uct_value}"
        # )

        return uct_value

    def _traverse(self, current_node):
        children = [
            outcome.search_node for outcome in self.outcomes if outcome.search_node.parent_trial == current_node.trial
        ]

        if not children:
            logger.info(f"MCT_SEARCH> Expanding childless node {current_node.trial}")
            return current_node

        if current_node.trial == _ROOT_TRIAL and len(children) < self.root_children:
            logger.info(f"MCT_SEARCH> Expanding root node {current_node.trial} with {len(children)} child(ren)")
            return current_node
        if len(children) < self.max_non_root_children:
            logger.info(f"MCT_SEARCH> Expanding non-root node {current_node.trial} with {len(children)} child(ren)")
            return current_node

        uct_values = [self._uct(child) for child in children]
        best_child = children[np.argmax(uct_values)]
        return self._traverse(best_child)

    def create_next_search_node(self, trial: int, operator_to_evolve: str, all_operator_names: list[str]) -> SearchNode:
        assert self.root_node is not None

        node_to_evolve = self._traverse(self.root_node)
        outcome_to_evolve = next(
            filter(lambda outcome: outcome.search_node.trial == node_to_evolve.trial, self.outcomes), None
        )
        assert outcome_to_evolve is not None
        return self._expand_tree(trial, operator_to_evolve, all_operator_names, outcome_to_evolve)
