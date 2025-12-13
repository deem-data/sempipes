import os
import traceback
from datetime import datetime
from typing import Any

import pandas as pd
import skrub
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.logging import get_logger
from sempipes.operators.operators import (
    ContextAwareMixin,
    EstimatorTransformer,
    OptimisableMixin,
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


_SYSTEM_PROMPT = (  # "You are an expert data scientist assistant helping data scientists write a data preprocessing pipeline for a predictive model. "
    "You are an expert data scientist assistant solving Kaggle problems. "
    "You answer only by generating code. Answer as concisely as possible."
)
_MAX_RETRIES = 5


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


def _add_memorized_history(
    memory: list[dict[str, Any]] | None,
    messages: list[dict[str, str]],
    target_metric: str,
) -> None:
    if memory is not None and len(memory) > 0:
        current_score = None

        for memory_line in memory:
            memorized_code = memory_line["update"]
            memorized_score = memory_line["score"]

            if current_score is None:
                improvement = abs(memorized_score)
            else:
                improvement = memorized_score - current_score

            if improvement > 0.0:
                add_feature_sentence = (
                    "The code was executed and improved the downstream performance. "
                    "You may choose to copy from this previous version of the code for the next version of the code."
                )
                current_score = memorized_score
            else:
                add_feature_sentence = (
                    f"The last code changes did not improve performance. " f"(Improvement: {improvement})"
                )

            messages += [
                {"role": "assistant", "content": memorized_code},
                {
                    "role": "user",
                    "content": f"Performance for last code block: {target_metric}={memorized_score:.5f}. "
                    f".{add_feature_sentence}\nNext codeblock:\n",
                },
            ]


def _try_to_execute(generated_code, left_df, left_join_key, right_df, right_join_key):
    agg_join_func = safe_exec(generated_code, variable_to_return="_sem_agg_join")

    # print("#" * 80)
    # print(generated_code)
    # print("#" * 80)

    left_sample = left_df.head(n=100)
    left_keys = left_sample[left_join_key].sample(frac=0.9, random_state=42)
    right_sample = right_df[right_df[right_join_key].isin(left_keys)]
    test_result = agg_join_func(left_join_key, left_sample, right_join_key, right_sample)

    if right_join_key != left_join_key and right_join_key in test_result.columns:
        test_result = test_result.drop(columns=[right_join_key])

    assert isinstance(test_result, pd.DataFrame)
    assert test_result.shape[0] == left_sample.shape[0]

    assert set(left_sample.columns).issubset(
        set(test_result.columns)
    ), f"Not all columns {left_sample.columns} from the left input are retained: {test_result.columns}"

    return test_result


class LLMCodeGenSemAggFeaturesEstimator(EstimatorTransformer, TransformerMixin, ContextAwareMixin, OptimisableMixin):  # pylint: disable=too-many-ancestors
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
        self.left_join_key = left_join_key
        self.right_join_key = right_join_key
        self.nl_prompt = nl_prompt
        self.how_many = how_many
        self.generated_code_: str | None = None

        self.eval_mode = eval_mode
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory
        self._inspirations: list[dict[str, Any]] | DataOp | None = _inspirations

    def empty_state(self):
        return {
            "generated_code": """
def _sem_agg_join(left_join_column, left_df, right_join_column, right_df):
    return left_df         
"""
        }

    def state_after_fit(self):
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self):
        if self.generated_code_ is not None:
            return self.generated_code_
        return OptimisableMixin.EMPTY_MEMORY_UPDATE

    def fit(self, stacked_inputs, y=None) -> "LLMCodeGenSemAggFeaturesEstimator":  # pylint: disable=unused-argument
        prompt_preview = self.nl_prompt[:40].replace("\n", " ").strip()

        if self._prefitted_state is not None:
            logger.debug(f"Using provided state for sempipes.sem_agg_features('{prompt_preview}...', {self.how_many})")
            self.generated_code_ = self._prefitted_state["generated_code"]
            return self

        samples = stacked_inputs["samples"]
        data_to_aggregate = stacked_inputs["data_to_aggregate"]

        logger.info(
            f"Fitting sempipes.sem_agg_features('{prompt_preview}...', {self.how_many}) on dataframes of shape {samples.shape} and {data_to_aggregate.shape} in mode '{self.eval_mode}'."
        )

        prompt = _build_prompt(
            samples,
            data_to_aggregate,
            self.left_join_key,
            self.right_join_key,
            self.nl_prompt,
            self.how_many,
            self._inspirations,
        )

        target_metric = "accuracy"
        if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
            target_metric = self._pipeline_summary.target_metric

        messages = []

        for attempt in range(1, _MAX_RETRIES + 1):
            if attempt == 1:
                messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                _add_memorized_history(self._memory, messages, target_metric)

            logger.debug("#" * 80)
            logger.debug(messages)
            logger.debug("#" * 80)

            code = generate_python_code_from_messages(messages)
            try:
                samples_copy = samples.copy(deep=True)
                data_to_aggregate_copy = data_to_aggregate.copy(deep=True)
                test_result = _try_to_execute(
                    code, samples_copy, self.left_join_key, data_to_aggregate_copy, self.right_join_key
                )
                new_columns = [column for column in test_result.columns if column not in samples.columns]

                logger.info(f"Computed {len(new_columns)} new feature columns: {new_columns}.")
                self.generated_code_ = code
                break

            except Exception as e:  # pylint: disable=broad-except
                logger.info(f"An error occurred in attempt {attempt}:", e)
                logger.debug(f"{e}", exc_info=True)

                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        + f"Code: ```python{code}```\n Retry and fix the errors!\n```python\n",
                    },
                ]

        if self.generated_code_ is None:
            logger.error(f"No code generated after {_MAX_RETRIES} retries. Falling back to empty state.")
            self.generated_code_ = self.empty_state()["generated_code"]

        return self

    def transform(self, stacked_inputs) -> pd.DataFrame:
        check_is_fitted(self, "generated_code_")
        samples = stacked_inputs["samples"]
        data_to_aggregate = stacked_inputs["data_to_aggregate"]

        num_samples_before = len(samples)
        # This is expensive, but we keep it for now to better understand operator failures.
        samples_copy_for_logging = samples.copy(deep=True)
        data_to_aggregate_copy_for_logging = data_to_aggregate.copy(deep=True)

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
            samples_copy_for_logging.to_csv(os.path.join(error_folder, "samples.csv"), index=False)
            data_to_aggregate_copy_for_logging.to_csv(os.path.join(error_folder, "data_to_aggregate.csv"), index=False)
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
