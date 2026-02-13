from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import dataclass_json
from skrub import DataOp

_ROOT_TRIAL = 0


@dataclass_json
@dataclass
class SearchNode:
    trial: int | None
    parent_trial: int | None
    memory: list[dict[str, Any]]
    predefined_state: dict[str, Any] | None
    parent_score: float | None
    inspirations: list[dict[str, Any]] = field(default_factory=list)


@dataclass_json
@dataclass
class Outcome:
    search_node: SearchNode
    state: dict[str, Any]
    score: float
    memory_update: str


class SearchPolicy(ABC):
    @abstractmethod
    def clone_empty(self) -> SearchPolicy:
        pass

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
    def create_next_search_node(self) -> SearchNode | None:
        """
        Create the next search node to explore.

        Returns:
            SearchNode: The next node to explore, or None if no more nodes can be generated.
        """

    @abstractmethod
    def get_outcomes(self) -> list[Outcome]:
        pass
