from abc import ABC, abstractmethod

class SemChooseOperator(ABC):
    @abstractmethod
    def set_params_on_estimator(self, data_op, estimator, choices, y=None):
        """Set parameters on the given estimator based on choices provided."""

class WithSemFeaturesOperator(ABC):
    @abstractmethod
    def generate_features_estimator(self, data_op, nl_prompt, name, how_many):
        """Return an estimator that computes features on a pandas df."""

class SemFillNAOperator(ABC):
    @abstractmethod
    def generate_imputation_estimator(self, data_op, target_column, nl_prompt):
        """Return an estimator that imputes missing values for the target column on a pandas df."""


class GyyreContextAwareMixin(ABC):
    gyyre_dag_summary: dict = {}

class GyyrePrefittableMixin(ABC):
    gyyre_prefitted_state: dict = {}

    @abstractmethod
    def state_after_fit(self):
        pass

class GyyreOptimisableMixin(ABC):
    gyyre_memory: list[dict] = {}

    @abstractmethod
    def memory_update_from_latest_fit(self):
        pass
