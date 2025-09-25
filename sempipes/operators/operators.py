from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# from skrub._data_ops._data_ops import Var
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from skrub import DataOp
from skrub.selectors._base import Filter

from sempipes.inspection.pipeline_summary import PipelineSummary


class ContextAwareMixin(ABC):
    """
    A mixin for sempipes operators that want to adjust themselves to the context in which they are used.

    The context is captured in a PipelineSummary object, which is provided to the operator as constructor parameter.
    The PipelineSummary contains information about the overall task, the model being used, the target variable, and the
    computational graph.

    Attributes:
        _pipeline_summary (PipelineSummary | None): A summary of the computational graph context.
    """

    _pipeline_summary: PipelineSummary | None = None


class PrefittableMixin(ABC):
    """
    A mixin for sempipes operators that can export and import their internal state for skipping the fit operation.

    The prefitted state is captured in a dict, which is provided to the operator as constructor parameter.
    This is required to enable context-aware optimisation, where the operator is fitted multiple times in
    different contexts, and where the pre-fitted state has to be evaluated repeatedly during cross-validation

    Attributes:
        _prefitted_state (dict[str, Any] | None): The prefitted state of the operator.
    """

    _prefitted_state: dict[str, Any] | DataOp | None = {}

    @abstractmethod
    def state_after_fit(self) -> dict[str, Any]:
        """
        Return the internal state of the operator after fitting, to be used for prefitting in future fits.

        Returns:
            dict[str, Any]: The internal state of the operator.
        """


class OptimisableMixin(PrefittableMixin):
    _memory: list[dict[str, Any]] | DataOp | None = {}

    @abstractmethod
    def empty_state(self) -> dict[str, Any]:
        """
        Return an empty internal state of the operator, to be used for prefitting in future fits.

        Returns:
            dict[str, Any]: The internal state of the operator.
        """

    @abstractmethod
    def memory_update_from_latest_fit(self) -> dict[str, Any]:
        pass


class EstimatorTransformer(BaseEstimator, TransformerMixin):
    pass


class OptimisableEstimatorTransformer(  # pylint: disable=too-many-ancestors
    EstimatorTransformer, ContextAwareMixin, OptimisableMixin, ABC
):
    pass


@dataclass(frozen=True)
class SemChoices:
    name: str
    params_and_prompts: dict[str, str]


class SemChooseOperator(ABC):
    @abstractmethod
    def set_params_on_estimator(
        self,
        estimator: BaseEstimator,
        choices: SemChoices,
        previous_results: list[DataFrame] | None = None,
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
    ) -> OptimisableEstimatorTransformer:
        """Return an estimator that computes features on a pandas df."""


class SemExtractFeaturesOperator(ABC):
    @abstractmethod
    def generate_features_extractor(
        self,
        nl_prompt: str,
        input_columns: list[str],
        output_columns: dict[str, str] | None,
    ) -> EstimatorTransformer:
        """Return an estimator that extracts features from the image/text on a pandas df."""


class SemFillNAOperator(ABC):
    @abstractmethod
    def generate_imputation_estimator(
        self,
        data_op: DataOp,
        target_column: str,
        nl_prompt: str,
    ) -> EstimatorTransformer:
        """Return an estimator that imputes missing values for the target column on a pandas df."""
