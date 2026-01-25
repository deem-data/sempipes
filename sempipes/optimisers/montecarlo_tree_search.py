import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name

from sempipes.optimisers.search_policy import _ROOT_TRIAL, Outcome, SearchNode, SearchPolicy


@dataclass
class SearchNodeStatistics:
    scores: list[float] = field(default_factory=list)

    def visits(self) -> int:
        return len(self.scores)


class MonteCarloTreeSearch(SearchPolicy):
    def __init__(
        self, 
        nodes_per_expansion=1, 
        c: float = 1.41, 
        root_children: int = 3, 
        max_non_root_children: int = 2
    ):
        self.c = c
        self.nodes_per_expansion = nodes_per_expansion
        self.root_children = root_children
        self.max_non_root_children = max_non_root_children
        self.min_score = float("inf")
        self.max_score = -float("inf")
        self.root_node: SearchNode | None = None
        self.outcomes: list[Outcome] = []
        self.search_node_stats: dict[int, SearchNodeStatistics] = {}

    def clone_empty(self) -> SearchPolicy:
        return MonteCarloTreeSearch(nodes_per_expansion=self.nodes_per_expansion, c=self.c, root_children=self.root_children, max_non_root_children=self.max_non_root_children)

    def create_root_node(self, dag_sink: DataOp, operator_name: str):
        data_op = find_node_by_name(dag_sink, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        print("\tMCT_SEARCH> Creating root node")
        root_node = SearchNode(
            trial=_ROOT_TRIAL,
            parent_trial=None,
            memory=[],
            predefined_state=empty_state,
            parent_score=None,
        )

        self.root_node = root_node
        self.search_node_stats[_ROOT_TRIAL] = SearchNodeStatistics()

        return root_node

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

        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)

        assert search_node.trial is not None
        if search_node.trial not in self.search_node_stats:
            self.search_node_stats[search_node.trial] = SearchNodeStatistics()
        self.search_node_stats[search_node.trial].scores.append(score)
        self._back_propagate(search_node, score)

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

        print((
            f"\t UCT of node {search_node.trial}: w_i: {w_i}, n_i: {n_i}, N_i: {N_i} ->"
            f"{(w_i / n_i)} + {self.c * np.sqrt(np.log(N_i) / n_i)} = {((w_i / n_i) + self.c * np.sqrt(np.log(N_i) / n_i))}"))

        return (w_i / n_i) + self.c * np.sqrt(np.log(N_i) / n_i)

    def _traverse(self, current_node):
        children = [
            outcome.search_node for outcome in self.outcomes if outcome.search_node.parent_trial == current_node.trial
        ]

        if not children:
            print(f"\t Expannding childless node {current_node.trial}")
            return current_node

        if current_node.trial == _ROOT_TRIAL and len(children) < self.root_children:
            print(f"\t Expanding root node {current_node.trial} with {len(children)} children")
            return current_node
        elif len(children) < self.max_non_root_children:
            print(f"\t Expanding non-root node {current_node.trial} with {len(children)} children")
            return current_node
        else:
            uct_values = [self._uct(child) for child in children]
            best_child = children[np.argmax(uct_values)]
            return self._traverse(best_child)

    def create_next_search_nodes(self) -> list[SearchNode]:
        assert self.root_node is not None

        node_to_evolve = self._traverse(self.root_node)
        corresponding_outcome = next(
            filter(lambda outcome: outcome.search_node.trial == node_to_evolve.trial, self.outcomes), None
        )
        assert corresponding_outcome is not None

        print(f"\tMCT_SEARCH> Trying to improve node with score {corresponding_outcome.score}")

        next_search_nodes = []
        for _ in range(self.nodes_per_expansion):
            updated_memory = copy.deepcopy(corresponding_outcome.search_node.memory)
            updated_memory.append({"update": corresponding_outcome.memory_update, "score": corresponding_outcome.score})
            next_node = SearchNode(
                trial=None,
                parent_trial=node_to_evolve.trial,
                memory=updated_memory,
                predefined_state=None,
                parent_score=corresponding_outcome.score,
            )
            next_search_nodes.append(next_node)
        return next_search_nodes

    def get_outcomes(self):
        return self.outcomes
