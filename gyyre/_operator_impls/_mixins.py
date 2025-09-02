from abc import ABC, abstractmethod


class GyyreContextAwareMixin(ABC):
    gyyre_dag_summary: dict = {}


class GyyrePrefittedMixin(ABC):
    gyyre_prefitted_state: dict = {}

    @abstractmethod
    def state_after_fit(self):
        pass


class GyyreMemoryMixin(ABC):
    gyyre_memory: list[dict] = {}

    @abstractmethod
    def memory_update_from_latest_fit(self):
        pass
