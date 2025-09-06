from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from skrub import DataOp
from skrub.selectors._base import Filter

from gyyre.optimisers._dag_summary import DagSummary


class GyyreContextAwareMixin(ABC):
    """
    A mixin for gyyre operators that want to adjust themselves to the context in which they are used.

    The context is captured in a DagSummary object, which is provided to the operator as constructor parameter.
    The DagSummary contains information about the overall task, the model being used, the target variable, and the
    computational graph.

    Attributes:
        gyyre_dag_summary (DagSummary | None): A summary of the computational graph context.
    """

    gyyre_dag_summary: DagSummary | None = None


class GyyrePrefittableMixin(ABC):
    """
    A mixin for gyyre operators that can export and import their internal state for skipping the fit operation.

    The prefitted state is captured in a dict, which is provided to the operator as constructor parameter.
    This is required to enable context-aware optimisation, where the operator is fitted multiple times in
    different contexts, and where the pre-fitted state has to be evaluated repeatedly during cross-validation

    Attributes:
        gyyre_prefitted_state (dict[str, Any] | None): The prefitted state of the operator.
    """

    gyyre_prefitted_state: dict[str, Any] = {}

    @abstractmethod
    def state_after_fit(self) -> dict[str, Any]:
        """
        Return the internal state of the operator after fitting, to be used for prefitting in future fits.

        Returns:
            dict[str, Any]: The internal state of the operator.
        """


class GyyreOptimisableMixin(ABC):
    gyyre_memory: list[dict[str, Any]] = {}

    @abstractmethod
    def memory_update_from_latest_fit(self) -> dict[str, Any]:
        pass


class SemChooseOperator(ABC):
    @abstractmethod
    def set_params_on_estimator(
        self,
        data_op: DataOp,
        estimator: BaseEstimator,
        choices,
        y=None,
    ) -> None:
        """Set parameters on the given estimator based on choices provided."""


class SemSelectOperator(ABC):
    @abstractmethod
    def generate_column_selector(
        self,
        nl_prompt: str,
    ) -> Filter:
        """Generate a column selector for dataframes."""


class WithSemFeaturesOperator(ABC):
    @abstractmethod
    def generate_features_estimator(
        self,
        data_op: DataOp,
        nl_prompt: str,
        name: str,
        how_many: int,
    ) -> BaseEstimator & TransformerMixin & GyyreContextAwareMixin & GyyrePrefittableMixin & GyyreOptimisableMixin:
        """Return an estimator that computes features on a pandas df."""


class SemFillNAOperator(ABC):
    @abstractmethod
    def generate_imputation_estimator(
        self,
        data_op: DataOp,
        target_column: str,
        nl_prompt: str,
    ) -> BaseEstimator & TransformerMixin:
        """Return an estimator that imputes missing values for the target column on a pandas df."""
