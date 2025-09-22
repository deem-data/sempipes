from __future__ import annotations

import copy
import random
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


class SearchStrategy(ABC):
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


class GreedySearch(SearchStrategy):
    def __init__(self):
        self.queue: list[SearchNode] = []
        self.outcomes: list[Outcome] = []
        self.search_node_under_evaluation: SearchNode | None = None
        self.recorded_op_state = None
        self.recorded_op_memory_update = None

    def initialize_search(self, dag_sink: DataOp, operator_name: str):
        data_op = find_node_by_name(dag_sink, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        initial_search_node = SearchNode(
            parent=None,
            memory=[],
            predefined_state=empty_state,
            parent_score=None,
        )

        self.queue.append(initial_search_node)

    def record_fit(self, learner, operator_name: str):
        op_fitted = learner.find_fitted_estimator(operator_name).transformer_
        self.recorded_op_state = op_fitted.state_after_fit()
        self.recorded_op_memory_update = op_fitted.memory_update_from_latest_fit()

        return self.recorded_op_state

    def record_score(self, score: float):
        search_node = self.search_node_under_evaluation
        assert search_node is not None

        outcome = Outcome(
            state=self.recorded_op_state,
            score=float(score),
            search_node=search_node,
        )
        self.outcomes.append(outcome)

        updated_memory = copy.deepcopy(search_node.memory)
        updated_memory.append({"update": self.recorded_op_memory_update, "score": float(score)})

        next_node = SearchNode(
            parent=search_node, memory=updated_memory, predefined_state=None, parent_score=float(score)
        )

        self.queue.append(next_node)
        self.recorded_op_state = None
        self.recorded_op_memory_update = None

    def next_search_node(self):
        assert len(self.queue) > 0
        search_node = self.queue.pop(0)
        self.search_node_under_evaluation = search_node
        return search_node

    def get_outcomes(self):
        return self.outcomes


class TreeSearch(SearchStrategy):
    def __init__(self):
        self.queue: list[SearchNode] = []
        self.outcomes: list[Outcome] = []
        self.search_node_under_evaluation: SearchNode | None = None
        self.recorded_op_state = None
        self.recorded_op_memory_update = None

    def initialize_search(self, dag_sink: DataOp, operator_name: str):
        data_op = find_node_by_name(dag_sink, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        initial_search_node = SearchNode(
            parent=None,
            memory=[],
            predefined_state=empty_state,
            parent_score=None,
        )

        self.queue.append(initial_search_node)

    def record_fit(self, learner, operator_name: str):
        op_fitted = learner.find_fitted_estimator(operator_name).transformer_
        self.recorded_op_state = op_fitted.state_after_fit()
        self.recorded_op_memory_update = op_fitted.memory_update_from_latest_fit()

        return self.recorded_op_state

    def record_score(self, score: float):
        search_node = self.search_node_under_evaluation
        assert search_node is not None

        outcome = Outcome(
            state=self.recorded_op_state,
            score=float(score),
            search_node=search_node,
        )
        self.outcomes.append(outcome)

        updated_memory = copy.deepcopy(search_node.memory)
        updated_memory.append({"update": self.recorded_op_memory_update, "score": float(score)})

        # TODO This should be done dynamically, we may run out of options in some cases
        for _ in range(3):
            next_node = SearchNode(
                parent=search_node, memory=updated_memory, predefined_state=None, parent_score=float(score)
            )

            self.queue.append(next_node)

        self.recorded_op_state = None
        self.recorded_op_memory_update = None

    def next_search_node(self):
        assert len(self.queue) > 0

        # TODO this greedily selects the best node to evolve, it should also do some exploration
        max_score = max(node.parent_score for node in self.queue)
        best_nodes = [node for node in self.queue if node.parent_score == max_score]
        next_node = random.choice(best_nodes)

        self.queue.remove(next_node)
        self.search_node_under_evaluation = next_node
        return next_node

    def get_outcomes(self):
        return self.outcomes
