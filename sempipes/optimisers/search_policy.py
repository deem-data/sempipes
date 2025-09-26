from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name


@dataclass
class SearchNode:
    parent: SearchNode | None
    memory: list[dict[str, Any]]
    predefined_state: dict[str, Any] | None
    parent_score: float | None


@dataclass
class Outcome:
    state: dict[str, Any]
    score: float
    search_node: SearchNode
    memory_update: str


class SearchPolicy(ABC):
    @abstractmethod
    def create_root_node(self, dag_sink: DataOp, operator_name: str):
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
    def create_next_search_node(self):
        pass

    @abstractmethod
    def get_outcomes(self):
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
            parent=None,
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

    def create_next_search_node(self):
        assert self.root_node is not None
        nodes_with_children = {id(outcome.search_node.parent) for outcome in self.outcomes}
        unprocessed_draft_nodes = [
            outcome.search_node
            for outcome in self.outcomes
            if id(outcome.search_node) not in nodes_with_children and outcome.search_node.parent is self.root_node
        ]
        num_unprocessed_draft_nodes = len(unprocessed_draft_nodes)

        if num_unprocessed_draft_nodes < self.min_num_drafts:  # pylint: disable=no-else-return
            print("\tTREE_SEARCH> Drafting new node")
            return self._draft()
        else:
            next_search_node = self._improve_best()
            print(f"\tTREE_SEARCH> Trying to improve node with score {next_search_node.parent_score}")
            return next_search_node

    def _draft(self):
        root_outcome = next(filter(lambda outcome: outcome.search_node is self.root_node, self.outcomes), None)
        assert root_outcome is not None

        updated_memory = copy.deepcopy(root_outcome.search_node.memory)
        updated_memory.append({"update": root_outcome.memory_update, "score": root_outcome.score})
        draft_node = SearchNode(
            parent=root_outcome.search_node,
            memory=updated_memory,
            predefined_state=None,
            parent_score=root_outcome.score,
        )

        return draft_node

    def _improve_best(self):
        best_outcome = max(self.outcomes, key=lambda outcome: outcome.score)

        updated_memory = copy.deepcopy(best_outcome.search_node.memory)
        updated_memory.append({"update": best_outcome.memory_update, "score": best_outcome.score})
        improve_node = SearchNode(
            parent=best_outcome.search_node,
            memory=updated_memory,
            predefined_state=None,
            parent_score=best_outcome.score,
        )

        return improve_node

    def get_outcomes(self):
        return self.outcomes
