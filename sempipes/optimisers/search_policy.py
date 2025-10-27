from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json
from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name

_ROOT_TRIAL = 0


@dataclass_json
@dataclass
class SearchNode:
    trial: int
    parent_trial: int | None
    memory: list[dict[str, Any]]
    predefined_state: dict[str, Any] | None
    parent_score: float | None


@dataclass_json
@dataclass
class Outcome:
    search_node: SearchNode
    state: dict[str, Any]
    score: float
    memory_update: str


class SearchPolicy(ABC):
    @abstractmethod
    def create_root_node(self, dag_sink: DataOp, operator_name: str) -> SearchNode:
        pass

    @abstractmethod
    def record_outcome(
        self,
        search_node: SearchNode,
        operator_state: dict[str, Any],
        score: float,
        operator_memory_update: str,
    ):
        pass

    @abstractmethod
    def create_next_search_node(self, trial: int) -> SearchNode:
        pass

    @abstractmethod
    def get_outcomes(self) -> list[Outcome]:
        pass


class TreeSearch(SearchPolicy):
    def __init__(self, min_num_drafts: int = 2):
        self.min_num_drafts = min_num_drafts
        self.root_node: SearchNode | None = None
        self.outcomes: list[Outcome] = []

    def create_root_node(self, dag_sink: DataOp, operator_name: str):
        data_op = find_node_by_name(dag_sink, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        print("\tTREE_SEARCH> Creating root node")
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

    def create_next_search_node(self, trial) -> SearchNode:
        assert self.root_node is not None
        nodes_with_children = {outcome.search_node.parent_trial for outcome in self.outcomes}
        unprocessed_draft_nodes = [
            outcome.search_node
            for outcome in self.outcomes
            if outcome.search_node.trial not in nodes_with_children and outcome.search_node.parent_trial is _ROOT_TRIAL
        ]
        num_unprocessed_draft_nodes = len(unprocessed_draft_nodes)

        if num_unprocessed_draft_nodes < self.min_num_drafts:  # pylint: disable=no-else-return
            print("\tTREE_SEARCH> Drafting new node")
            return self._draft(trial)
        else:
            next_search_node = self._improve_best(trial)
            print(f"\tTREE_SEARCH> Trying to improve node with score {next_search_node.parent_score}")
            return next_search_node

    def _draft(self, trial: int) -> SearchNode:
        root_outcome = next(filter(lambda outcome: outcome.search_node is self.root_node, self.outcomes), None)
        assert root_outcome is not None

        updated_memory = copy.deepcopy(root_outcome.search_node.memory)
        updated_memory.append({"update": root_outcome.memory_update, "score": root_outcome.score})
        draft_node = SearchNode(
            trial=trial,
            parent_trial=root_outcome.search_node.trial,
            memory=updated_memory,
            predefined_state=None,
            parent_score=root_outcome.score,
        )

        return draft_node

    def _improve_best(self, trial: int) -> SearchNode:
        best_outcome = max(self.outcomes, key=lambda outcome: outcome.score)

        updated_memory = copy.deepcopy(best_outcome.search_node.memory)
        updated_memory.append({"update": best_outcome.memory_update, "score": best_outcome.score})
        improve_node = SearchNode(
            trial=trial,
            parent_trial=best_outcome.search_node.trial,
            memory=updated_memory,
            predefined_state=None,
            parent_score=best_outcome.score,
        )

        return improve_node

    def get_outcomes(self):
        return self.outcomes
