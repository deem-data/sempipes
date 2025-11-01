from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    def create_next_search_nodes(self) -> list[SearchNode]:
        pass

    @abstractmethod
    def get_outcomes(self) -> list[Outcome]:
        pass
