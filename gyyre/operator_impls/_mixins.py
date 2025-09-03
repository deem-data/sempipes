from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


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
