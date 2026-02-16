# This code is based on Apache-licensed code from https://github.com/noahho/CAAFE/
# TODO This class needs some serious refactoring
import os
import traceback
from datetime import datetime
from typing import Any

import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.logging import get_logger
from sempipes.operators.operators import (
    ContextAwareMixin,
    OptimisableMixin,
    SemGenFeaturesOperator,
)

logger = get_logger()

_MAX_RETRIES = 10
_SYSTEM_PROMPT = (
    "You are an expert datascientist assistant solving Kaggle problems. "
    "You answer only by generating code. Answer as concisely as possible."
)


def _get_prompt(  # pylint: disable=too-many-locals
    df: pd.DataFrame,
    nl_prompt: str,
    how_many: int,
    samples: str | None = None,
    pipeline_summary: PipelineSummary | None = None,
    inspirations: list[dict[str, Any]] | None = None,
) -> str:
    data_description_unparsed = None

    task_description = (
        "This code generates additional columns that are useful for a downstream classification "
        "algorithm (such as XGBoost) predicting a target label."
    )
    usefulness = ""
    model_reference = "classifier"
    target_metric = "accuracy"

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

    if pipeline_summary is not None:
        task_type = pipeline_summary.task_type
        model = pipeline_summary.model
        target_name = pipeline_summary.target_name
        data_description_unparsed = pipeline_summary.dataset_description

        target_description = ""
        if pipeline_summary.target_description:
            target_description = f" ({pipeline_summary.target_description})"
        if task_type and model and target_name:
            task_description = (
                f"This code generates additional columns that are useful for a "
                f'downstream {task_type} algorithm ({model}) predicting "{target_name} {target_description}".'
            )

        if pipeline_summary.target_metric:
            target_metric = pipeline_summary.target_metric

        if task_type and target_name:
            action = "predict"
            if task_type == "classification":
                action = "classify"
            usefulness = (
                f"\n# Usefulness: (Description why this adds useful real world knowledge "
                f'to {action} "{target_name}" according to dataset description and attributes.)'
            )

        if task_type == "regression":
            model_reference = "regressor"

    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

This code was written by an expert datascientist working to improve predictions. Number of samples (rows) in training dataset: {int(len(df))}

{task_description}
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g.
be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of
columns closely and consider the datatypes and meanings of classes.
The {model_reference} will be trained on the dataset with the generated columns and evaluated on a holdout set. The
evaluation metric is {target_metric}. The best performing code will be selected.

The data scientist wants you to take special care of the following: {nl_prompt}.

{inspiration_examples}

Make sure that the code produces exactly the same columns when applied to a new dataframe with the same input columns.

Generate a Python function called `_sem_gen_features` that takes a single argument `df` (a pandas DataFrame) and
returns the modified DataFrame with up to {how_many} new feature columns added. Generate as many features as useful
for the downstream {model_reference}, but as few as necessary to reach good performance.{usefulness}

DO NOT INCLUDE EXAMPLE USAGE CODE. WRAP YOUR RESPONSE CODE IN ```python and ```.

MAKE SURE THAT THE NEW COLUMNS HAVE MEANINGFUL NAMES.

EXPLAIN YOUR RATIONALE FOR CHOOSING FEATURES IN COMMENTS IN THE PYTHON CODE. For each newly generated
feature, add a comment to the code that describes the feature, explains why you chose it and why this feature adds
useful real world knowledge for the downstream model.
"""


def _build_prompt_from_df(
    df: pd.DataFrame,
    nl_prompt: str,
    how_many: int,
    pipeline_summary: PipelineSummary | None = None,
    inspirations: list[dict[str, Any]] | None = None,
) -> str:
    samples = ""
    df_ = df.head(10)
    for column in list(df_):
        null_ratio = df[column].isna().mean()
        nan_freq = f"{null_ratio * 100:.2g}"
        sampled_values = df_[column].tolist()
        if str(df[column].dtype) == "float64":
            sampled_values = [float(round(sample, 2)) for sample in sampled_values]
        samples += f"{df_[column].name} ({df[column].dtype}): NaN-freq [{nan_freq}%], Samples {sampled_values}\n"
    return _get_prompt(
        df, nl_prompt, how_many, samples=samples, pipeline_summary=pipeline_summary, inspirations=inspirations
    )


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


def _try_to_execute(df: pd.DataFrame, generated_code: str) -> pd.DataFrame:
    df_sample = df.head(100).copy(deep=True)
    columns_before = set(df_sample.columns)

    gen_features_func = safe_exec(generated_code, variable_to_return="_sem_gen_features")
    result = gen_features_func(df_sample)

    assert isinstance(result, pd.DataFrame), "_sem_gen_features must return a DataFrame"
    assert result.shape[0] == df_sample.shape[0], "_sem_gen_features must not change number of rows"
    assert columns_before.issubset(
        set(result.columns)
    ), f"Not all original columns retained: missing {columns_before - set(result.columns)}"

    new_columns = sorted(set(result.columns) - columns_before)
    logger.info(f"Computed {len(new_columns)} new feature columns: {new_columns}")

    return result


# pylint: disable=too-many-ancestors
class LLMFeatureGenerator(BaseEstimator, TransformerMixin, ContextAwareMixin, OptimisableMixin):
    def __init__(
        self,
        nl_prompt: str,
        how_many: int,
        eval_mode: str = skrub.eval_mode(),
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
    ) -> None:
        self.nl_prompt = nl_prompt
        self.how_many = how_many
        self.eval_mode = eval_mode
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory
        self._inspirations: list[dict[str, Any]] | DataOp | None = _inspirations
        self.generated_code_: str | None = None

    def empty_state(self):
        return {
            "generated_code": """
def _sem_gen_features(df):
    return df
"""
        }

    def state_after_fit(self):
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self):
        if self.generated_code_ is not None:
            return self.generated_code_
        return OptimisableMixin.EMPTY_MEMORY_UPDATE

    def fit(self, df: pd.DataFrame, y=None, **fit_params):  # pylint: disable=unused-argument
        prompt_preview = self.nl_prompt[:40].replace("\n", " ").strip()

        if self._prefitted_state is not None:
            logger.debug(f"Using provided state for sempipes.sem_gen_features('{prompt_preview}...', {self.how_many})")
            self.generated_code_ = self._prefitted_state["generated_code"]
            return self

        logger.info(
            f"Fitting sempipes.sem_gen_features('{prompt_preview}...', {self.how_many}) on dataframe of shape {df.shape} in mode '{self.eval_mode}'."
        )

        prompt = _build_prompt_from_df(df, self.nl_prompt, self.how_many, self._pipeline_summary, self._inspirations)

        target_metric = "accuracy"
        if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
            target_metric = self._pipeline_summary.target_metric

        messages = []

        for attempt in range(1, _MAX_RETRIES + 1):
            if attempt == 1:
                messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                _add_memorized_history(self._memory, messages, target_metric)

            code = generate_python_code_from_messages(messages)
            try:
                _try_to_execute(df.copy(deep=True), code)

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

    def transform(self, df):
        check_is_fitted(self, "generated_code_")
        df_copy_for_logging = df.copy(deep=True)

        try:
            gen_features_func = safe_exec(self.generated_code_, variable_to_return="_sem_gen_features")
            df = gen_features_func(df.copy(deep=True))
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            error_folder = f".sem_gen_features_error_{timestamp}"
            os.makedirs(error_folder, exist_ok=True)
            df_copy_for_logging.to_csv(os.path.join(error_folder, "input_df.csv"), index=False)
            if self.generated_code_ is not None:
                with open(os.path.join(error_folder, "executed_code.py"), "w", encoding="utf-8") as f:
                    f.write(self.generated_code_)
            stack_trace_file_path = os.path.join(error_folder, "stack_trace.txt")
            with open(stack_trace_file_path, "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
            logger.error(f"Error occurred in transform: {e}", exc_info=True)
            raise e

        return df


class SemGenFeaturesCaafe(SemGenFeaturesOperator):
    def generate_features_estimator(self, data_op: DataOp, nl_prompt: str, name: str, how_many: int):
        _pipeline_summary = skrub.var(f"sempipes_pipeline_summary__{name}", None)
        _prefitted_state = skrub.var(f"sempipes_prefitted_state__{name}", None)
        _memory = skrub.var(f"sempipes_memory__{name}", [])
        _inspirations = skrub.var(f"sempipes_inspirations__{name}", [])

        return LLMFeatureGenerator(
            nl_prompt,
            how_many,
            _pipeline_summary=_pipeline_summary,
            _prefitted_state=_prefitted_state,
            _memory=_memory,
            _inspirations=_inspirations,
        )


def sem_gen_features(
    self: DataOp,
    nl_prompt: str,
    name: str,
    how_many: int = 10,
) -> DataOp:
    data_op = self
    feature_gen_estimator = SemGenFeaturesCaafe().generate_features_estimator(data_op, nl_prompt, name, how_many)
    return self.skb.apply(feature_gen_estimator, how="no_wrap").skb.set_name(name)
