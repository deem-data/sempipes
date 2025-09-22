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
    memory_update: dict[str, Any] | None


class SearchPolicy(ABC):
    @abstractmethod
    def initialize_search(self, dag_sink: DataOp, operator_name: str):
        pass

    @abstractmethod
    def record_fit(self, learner, operator_name: str):
        pass

    @abstractmethod
    def record_score(self, score: float):
        pass

    @abstractmethod
    def next_search_node(self):
        pass

    @abstractmethod
    def get_outcomes(self):
        pass


class TreeSearch(SearchPolicy):
    def __init__(self, min_num_drafts: int = 2):
        self.min_num_drafts = min_num_drafts
        self.search_node_to_evaluate: SearchNode | None = None
        self.root_node: SearchNode | None = None
        self.outcomes: list[Outcome] = []
        self.recorded_op_state: dict[str, Any] | None = None
        self.recorded_op_memory_update = None

    def initialize_search(self, dag_sink: DataOp, operator_name: str):
        data_op = find_node_by_name(dag_sink, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        print("\tTREE_SEARCH> Creating root node")
        initial_search_node = SearchNode(
            parent=None,
            memory=[],
            predefined_state=empty_state,
            parent_score=None,
        )

        self.root_node = initial_search_node
        self.search_node_to_evaluate = initial_search_node

    def record_fit(self, learner, operator_name: str):
        op_fitted = learner.find_fitted_estimator(operator_name).transformer_
        self.recorded_op_state = op_fitted.state_after_fit()
        self.recorded_op_memory_update = op_fitted.memory_update_from_latest_fit()

        return self.recorded_op_state

    def record_score(self, score: float):
        search_node = self.search_node_to_evaluate
        assert search_node is not None
        assert self.recorded_op_state is not None

        outcome = Outcome(
            state=self.recorded_op_state,  # type: ignore[arg-type]
            score=score,
            search_node=search_node,
            memory_update=self.recorded_op_memory_update,
        )
        self.outcomes.append(outcome)
        self.recorded_op_state = None
        self.recorded_op_memory_update = None

        self._schedule_next_node()

    def _schedule_next_node(self):
        # TODO Rewrite and add asserts
        assert self.root_node is not None
        nodes_with_children = {id(outcome.search_node.parent) for outcome in self.outcomes}
        unprocessed_draft_nodes = [
            outcome.search_node
            for outcome in self.outcomes
            if id(outcome.search_node) not in nodes_with_children and outcome.search_node.parent is self.root_node
        ]

        if len(unprocessed_draft_nodes) < self.min_num_drafts:
            print("\tTREE_SEARCH> Drafting new node")
            self.search_node_to_evaluate = self._draft()
        else:
            self.search_node_to_evaluate = self._improve_best()
            print(f"\tTREE_SEARCH> Trying to improve node with score {self.search_node_to_evaluate.parent_score}")

    def _draft(self):
        # TODO Rewrite and add asserts
        root_outcome = [outcome for outcome in self.outcomes if outcome.search_node is self.root_node][0]

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

    def next_search_node(self):
        assert self.search_node_to_evaluate is not None
        return self.search_node_to_evaluate

    def get_outcomes(self):
        return self.outcomes
