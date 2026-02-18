import os
import traceback
from datetime import datetime
from typing import Any

import pandas as pd
import skrub
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.logging import get_logger
from sempipes.operators.iterative_code_gen_base import IterativeCodeGenEstimator
from sempipes.operators.operators import (
    EstimatorTransformer,
    SemAggFeaturesOperator,
)

logger = get_logger()


def _dataframe_mini_summary(df: pd.DataFrame, sample_size: int = 10) -> str:
    summary_lines = []

    for column in df.columns:
        column_type = df[column].dtype
        missing_ratio = df[column].isna().mean()
        sample_values = (
            df[column].dropna().sample(n=min(sample_size, df[column].dropna().shape[0]), random_state=42).tolist()
            if df[column].notna().any()
            else []
        )

        summary_lines.append(
            f"Column: {column}\n"
            f"  Type: {column_type}\n"
            f"  Missing ratio: {missing_ratio:.2%}\n"
            f"  Sample values: {sample_values}\n"
        )

    return "\n".join(summary_lines)


def _build_prompt(left_df, right_df, left_join_column, right_join_column, nl_prompt, how_many, inspirations):  # pylint: disable=too-many-positional-arguments
    left_df_summary = _dataframe_mini_summary(left_df)
    right_df_summary = _dataframe_mini_summary(right_df)

    inspiration_examples = ""
    if inspirations and len(inspirations) > 0:
        inspiration_examples += (
            "Here are some examples of code that has been used to generate features in the past, together with the resulting scores."
            "You can use these as inspiration to generate your own code.\n\n"
        )

        # Sort inspirations by score in descending order and select top 3
        top_inspirations = sorted(inspirations, key=lambda x: x["score"], reverse=True)[:3]

        for i, inspiration in enumerate(top_inspirations):
            code = inspiration["state"]["generated_code"]
            score = inspiration["score"]
            inspiration_examples += f"Example {i+1}:\n```python\n{code}\n```\nScore: {score:.4f}\n\n"

    return f"""
        You need to extend a data preparation pipeline for a machine learning model with generating additional features for the training data. The code already has a dataframe with
        existing features and the goal is to left join another dataframe with the existing dataframe to generate more features. For that, you need
        to decide which columns to include from the dataframe to join and how to aggregate them in a way that helps the downstream model. A single column can be
        included multiple times with different aggregations.

        The dataframe with the existing training data looks as follows:

        {left_df_summary}

        The `left_join_column` is: {left_join_column}

        The dataframe to left join and aggregate looks as follows:

        {right_df_summary}

        The `right_join_column` is: {right_join_column}

        Here is the full output of df.describe(include='all') for the dataframe to join. You can use these statistics
        as constants for your aggregation functions if needed.

        {right_df.describe(include='all').to_string()}

        Please take special care of the following:

        {nl_prompt}

        {inspiration_examples}

        Generate a Python function called `_sem_agg_join` that takes four arguments: `left_join_column`, `left_df`, `right_join_column` and `right_df`
        and conducts the desired left join and aggregations. Your code should generate {how_many} new features.

        DO NOT INCLUDE EXAMPLE USAGE CODE. WRAP YOUR RESPONSE CODE IN ```python and ```.

        MAKE SURE THAT THE NEW COLUMNS HAVE MEANINGFUL NAMES.

        EXPLAIN YOUR RATIONALE FOR CHOOSING AGGREGATION FUNCTIONS IN COMMENTS IN THE PYTHON CODE. For each newly generated
        feature, add a comment to the code that describes the features, explains why you chose it and why this feature adds useful real world knowledge for the downstream model.
    """


def _validate_generated_code(generated_code, left_df, left_join_key, right_df, right_join_key):
    agg_join_func = safe_exec(generated_code, variable_to_return="_sem_agg_join")

    left_sample = left_df.head(n=100).copy(deep=True)
    left_keys = left_sample[left_join_key].sample(frac=0.9, random_state=42)
    right_sample = right_df[right_df[right_join_key].isin(left_keys)].copy(deep=True)
    test_result = agg_join_func(left_join_key, left_sample, right_join_key, right_sample)

    if right_join_key != left_join_key and right_join_key in test_result.columns:
        test_result = test_result.drop(columns=[right_join_key])

    assert isinstance(test_result, pd.DataFrame)
    assert test_result.shape[0] == left_sample.shape[0]

    assert set(left_sample.columns).issubset(
        set(test_result.columns)
    ), f"Not all columns {left_sample.columns} from the left input are retained: {test_result.columns}"

    return test_result


class LLMCodeGenSemAggFeaturesEstimator(IterativeCodeGenEstimator):  # pylint: disable=too-many-ancestors
    _SYSTEM_PROMPT = (
        "You are an expert data scientist assistant solving Kaggle problems. "
        "You answer only by generating code. Answer as concisely as possible."
    )

    def __init__(
        self,
        left_join_key: str,
        right_join_key: str,
        nl_prompt: str,
        how_many: int,
        eval_mode: str = skrub.eval_mode(),
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
    ):
        super().__init__(nl_prompt, _pipeline_summary, _prefitted_state, _memory, _inspirations)
        self.left_join_key = left_join_key
        self.right_join_key = right_join_key
        self.how_many = how_many
        self.eval_mode = eval_mode

    def empty_state(self):
        return {
            "generated_code": """
def _sem_agg_join(left_join_column, left_df, right_join_column, right_df):
    return left_df
"""
        }

    def _try_to_execute(self, code, samples, data_to_aggregate):  # pylint: disable=arguments-differ
        test_result = _validate_generated_code(
            code, samples, self.left_join_key, data_to_aggregate, self.right_join_key
        )
        new_columns = [column for column in test_result.columns if column not in samples.columns]
        logger.info(f"Computed {len(new_columns)} new feature columns: {new_columns}.")

    def fit(self, stacked_inputs, y=None) -> "LLMCodeGenSemAggFeaturesEstimator":  # pylint: disable=unused-argument
        if self._load_prefitted_state():
            return self

        samples = stacked_inputs["samples"]
        data_to_aggregate = stacked_inputs["data_to_aggregate"]

        prompt = _build_prompt(
            samples,
            data_to_aggregate,
            self.left_join_key,
            self.right_join_key,
            self.nl_prompt,
            self.how_many,
            self._inspirations,
        )

        self._iterative_code_generation(samples, data_to_aggregate, prompt=prompt)
        return self

    def transform(self, stacked_inputs) -> pd.DataFrame:
        check_is_fitted(self, "generated_code_")
        samples = stacked_inputs["samples"]
        data_to_aggregate = stacked_inputs["data_to_aggregate"]

        num_samples_before = len(samples)

        try:
            # We have to copy the inputs, as some generated code might modify the data in-place.
            samples_copy = samples.copy(deep=True)
            data_to_aggregate_copy = data_to_aggregate.copy(deep=True)

            agg_join_func = safe_exec(self.generated_code_, variable_to_return="_sem_agg_join")  # type: ignore
            result_df = agg_join_func(self.left_join_key, samples_copy, self.right_join_key, data_to_aggregate_copy)

        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            error_folder = f".sem_agg_features_error_{timestamp}"
            os.makedirs(error_folder, exist_ok=True)
            if self.generated_code_ is not None:
                with open(os.path.join(error_folder, "executed_code.py"), "w", encoding="utf-8") as f:
                    f.write(self.generated_code_)
            stack_trace_file_path = os.path.join(error_folder, "stack_trace.txt")
            with open(stack_trace_file_path, "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
            logger.error(f"Error occurred in transform: {e}", exc_info=True)
            raise e

        if self.right_join_key in result_df.columns:
            result_df = result_df.drop(columns=[self.right_join_key])

        num_samples_after = len(result_df)
        assert num_samples_before == num_samples_after

        return result_df


def sem_agg_features(  # pylint: disable=too-many-positional-arguments
    self: DataOp,
    right_data_op: DataOp,
    left_on: str,
    right_on: str,
    nl_prompt: str,
    name: str,
    how_many: int = 10,
) -> DataOp:
    left_data_op = self

    inputs = skrub.as_data_op({"samples": left_data_op, "data_to_aggregate": right_data_op}).skb.set_name(
        f"{name}__inputs"
    )

    _pipeline_summary = skrub.var(f"sempipes_pipeline_summary__{name}", None)
    _prefitted_state = skrub.var(f"sempipes_prefitted_state__{name}", None)
    _memory = skrub.var(f"sempipes_memory__{name}", [])
    _inspirations = skrub.var(f"sempipes_inspirations__{name}", [])

    agg_joiner = LLMCodeGenSemAggJoinFeaturesOperator().generate_agg_join_features_estimator(
        left_join_key=left_on,
        right_join_key=right_on,
        nl_prompt=nl_prompt,
        how_many=how_many,
        eval_mode=skrub.eval_mode(),
        _pipeline_summary=_pipeline_summary,
        _prefitted_state=_prefitted_state,
        _memory=_memory,
        _inspirations=_inspirations,
    )

    return inputs.skb.apply(agg_joiner).skb.set_name(name)


class LLMCodeGenSemAggJoinFeaturesOperator(SemAggFeaturesOperator):
    def generate_agg_join_features_estimator(
        self,
        left_join_key: str,
        right_join_key: str,
        nl_prompt: str,
        how_many: int,
        eval_mode: str = skrub.eval_mode(),
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
    ) -> EstimatorTransformer:
        return LLMCodeGenSemAggFeaturesEstimator(
            left_join_key=left_join_key,
            right_join_key=right_join_key,
            nl_prompt=nl_prompt,
            how_many=how_many,
            eval_mode=eval_mode,
            _pipeline_summary=_pipeline_summary,
            _prefitted_state=_prefitted_state,
            _memory=_memory,
            _inspirations=_inspirations,
        )
