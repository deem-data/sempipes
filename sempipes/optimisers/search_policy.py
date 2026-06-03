from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import dataclass_json
from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name

from sempipes.logging import get_logger

logger = get_logger()

_ROOT_TRIAL = 0


@dataclass_json
@dataclass
class OperatorStates:
    states_per_operator: dict[str, dict[str, Any]] = field(default_factory=dict)

    def operators(self) -> list[str]:
        return list(self.states_per_operator.keys())

    def exists_for(self, operator_name: str) -> bool:
        return operator_name in self.states_per_operator

    def get(self, operator_name: str) -> dict[str, Any]:
        return self.states_per_operator[operator_name]

    def set(self, operator_name: str, state: dict[str, Any]):
        self.states_per_operator[operator_name] = state


@dataclass_json
@dataclass
class OperatorMemories:
    memories_per_operator: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def exists_for(self, operator_name: str) -> bool:
        return operator_name in self.memories_per_operator

    def append(self, operator_name: str, memory: dict[str, Any]):
        if operator_name not in self.memories_per_operator:
            self.memories_per_operator[operator_name] = []
        self.memories_per_operator[operator_name].append(memory)

    def get(self, operator_name: str) -> list[dict[str, Any]]:
        if operator_name not in self.memories_per_operator:
            return []

        return self.memories_per_operator[operator_name]

    def set(self, operator_name: str, memories: list[dict[str, Any]]):
        self.memories_per_operator[operator_name] = memories


@dataclass_json
@dataclass
class SearchNode:
    trial: int
    parent_trial: int | None = None
    operator_to_evolve: str | None = None
    operators_evolved: list[str] = field(default_factory=list)
    memories: OperatorMemories = field(default_factory=OperatorMemories)
    fixed_states: OperatorStates = field(default_factory=OperatorStates)
    parent_score: float | None = None
    inspirations: OperatorMemories = field(default_factory=OperatorMemories)


@dataclass_json
@dataclass
class Outcome:
    search_node: SearchNode
    states: OperatorStates
    score: float
    memory_updates: dict[str, str]


class SearchPolicy(ABC):
    def __init__(self):
        self.root_node: SearchNode | None = None
        self.outcomes: list[Outcome] = []

    def create_root_node(self, dag_sink: DataOp, all_operator_names: list[str]) -> SearchNode:
        empty_states = OperatorStates()

        for operator_name in all_operator_names:
            data_op = find_node_by_name(dag_sink, operator_name)
            assert data_op is not None
            assert data_op._skrub_impl is not None
            assert data_op._skrub_impl.estimator is not None
            empty_states.set(operator_name, data_op._skrub_impl.estimator.empty_state())

        logger.info("COLOPRO> Creating root node")
        root_node = SearchNode(trial=_ROOT_TRIAL, fixed_states=empty_states)

        self.root_node = root_node
        return root_node

    def record_outcome(
        self,
        search_node: SearchNode,
        operator_states: dict[str, dict[str, Any]] | None,
        score: float,
        operator_memory_updates: dict[str, str],
    ):
        states = copy.deepcopy(search_node.fixed_states)
        if operator_states is not None:
            for operator_to_evolve in operator_states:
                states.set(operator_to_evolve, operator_states[operator_to_evolve])

        outcome = Outcome(
            states=states,
            score=score,
            search_node=search_node,
            memory_updates=operator_memory_updates,
        )
        self.outcomes.append(outcome)

    def get_outcomes(self):
        return self.outcomes

    def _expand_tree(
        self,
        trial: int,
        operator_to_evolve: str,
        all_operator_names: list[str],
        outcome_to_evolve: Outcome,
    ) -> SearchNode:
        fixed_operators = [name for name in all_operator_names if name != operator_to_evolve]
        fixed_operator_states = OperatorStates()
        for operator_name in fixed_operators:
            assert outcome_to_evolve.states.exists_for(operator_name)  # type: ignore[attr-defined]
            fixed_operator_states.set(operator_name, outcome_to_evolve.states.get(operator_name))  # type: ignore[attr-defined]

        logger.info(f"COLOPRO> Trying to improve node with score {outcome_to_evolve.score}")
        updated_memories = copy.deepcopy(outcome_to_evolve.search_node.memories)

        for evolved_operator in outcome_to_evolve.memory_updates:
            updated_memories.append(
                evolved_operator,
                {"update": outcome_to_evolve.memory_updates[evolved_operator], "score": outcome_to_evolve.score},
            )

        next_node = SearchNode(
            trial=trial,
            parent_trial=outcome_to_evolve.search_node.trial,
            operator_to_evolve=operator_to_evolve,
            memories=updated_memories,
            fixed_states=fixed_operator_states,
            parent_score=outcome_to_evolve.score,
        )

        return next_node

    @abstractmethod
    def clone_empty(self) -> SearchPolicy:
        pass

    @abstractmethod
    def create_next_search_node(self, trial: int, operator_to_evolve: str, all_operator_names: list[str]) -> SearchNode:
        pass
