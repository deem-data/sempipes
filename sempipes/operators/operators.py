from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
    EMPTY_MEMORY_UPDATE: str = ""

    _memory: list[dict[str, Any]] | DataOp | None = {}

    @abstractmethod
    def empty_state(self) -> dict[str, Any]:
        """
        Return an empty internal state of the operator, to be used for prefitting in future fits.

        Returns:
            dict[str, Any]: The internal state of the operator.
        """

    @abstractmethod
    def memory_update_from_latest_fit(self) -> str:
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


class SemGenFeaturesOperator(ABC):
    @abstractmethod
    def generate_features_estimator(
        self,
        data_op: DataOp,
        nl_prompt: str,
        name: str,
        how_many: int,
    ) -> OptimisableEstimatorTransformer:
        """Return an estimator that computes features on a pandas df."""


class SemAggFeaturesOperator(ABC):
    @abstractmethod
    def generate_agg_join_features_estimator(
        self,
        left_join_key: str,
        right_join_key: str,
        nl_prompt: str,
        how_many: int,
    ) -> EstimatorTransformer:
        """Return an estimator that computes features via a left join aggregation query on two pandas dataframes."""


class SemExtractFeaturesOperator(ABC):
    @abstractmethod
    def generate_features_extractor(
        self,
        nl_prompt: str,
        input_columns: list[str],
        output_columns: dict[str, str] | None,
        **kwargs,
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


class SemRefineOperator(ABC):
    @abstractmethod
    def generate_refinement_estimator(
        self,
        data_op: DataOp,
        target_column: str,
        nl_prompt: str,
        refine_with_existing_values_only: bool,
        name: str,
    ) -> EstimatorTransformer:
        """Return an estimator that refines values in the target column on a pandas df."""


class SemCleanOperator(ABC):
    @abstractmethod
    def generate_cleaning_estimator(
        self, data_op: DataOp, nl_prompt: str, columns: list[str], name: str
    ) -> EstimatorTransformer:
        """Return an estimator that applies classical tabular data cleaning techniques to the selected columns of a pandas df."""


class SemAugmentDataOperator(ABC):
    @abstractmethod
    def generate_data_generator(
        self,
        data_op: DataOp,
        nl_prompt: str,
        name: str,
        number_of_rows_to_generate: int,
        **kwargs,
    ) -> EstimatorTransformer:
        """Return an estimator that generates a new set of the data."""


class SemDistillDataOperator(ABC):
    @abstractmethod
    def generate_data_distiller(self, nl_prompt: str, number_of_rows: int) -> TransformerMixin:
        """Return an estimator that distills the data."""
