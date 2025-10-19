# This code is based on Apache-licensed code from https://github.com/noahho/CAAFE/
# TODO This class needs some serious cleanup / refactoring
from typing import Any

import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.operators.operators import (
    ContextAwareMixin,
    OptimisableMixin,
    WithSemFeaturesOperator,
)

_MAX_RETRIES = 5
_SYSTEM_PROMPT = (
    "You are an expert datascientist assistant solving Kaggle problems. "
    "You answer only by generating code. Answer as concisely as possible."
)


def _get_prompt(
    df: pd.DataFrame,
    nl_prompt: str,
    how_many: int,
    samples: str | None = None,
    pipeline_summary: PipelineSummary | None = None,
) -> str:
    data_description_unparsed = None

    task_description = (
        "This code generates additional columns that are useful for a downstream classification "
        "algorithm (such as XGBoost) predicting a target label."
    )
    usefulness = ""
    model_reference = "classifier"
    target_metric = "accuracy"

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

This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds 
new columns to the dataset. Number of samples (rows) in training dataset: {int(len(df))}

{task_description}
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. 
be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of 
columns closely and consider the datatypes and meanings of classes.
The {model_reference} will be trained on the dataset with the generated columns and evaluated on a holdout set. The 
evaluation metric is {target_metric}. The best performing code will be selected.
Added columns can be used in other codeblocks.

The data scientist wants you to take special care to the following: {nl_prompt}.

Make sure that the code produces exactly the same columns when applied to a new dataframe with the same input columns.

Code formatting for each added column:
```python
# (Feature name and description){usefulness}
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}':
{list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Each codeblock generates up to {how_many} useful columns. Generate as many features as useful for downstream 
{model_reference}, but as few as necessary to reach good performance.
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""


def _build_prompt_from_df(
    df: pd.DataFrame, nl_prompt: str, how_many: int, pipeline_summary: PipelineSummary | None = None
) -> str:
    samples = ""
    df_ = df.head(10)
    for column in list(df_):
        null_ratio = df[column].isna().mean()
        nan_freq = f"{null_ratio * 100:.2g}"
        sampled_values = df_[column].tolist()
        if str(df[column].dtype) == "float64":
            sampled_values = [round(sample, 2) for sample in sampled_values]
        samples += f"{df_[column].name} ({df[column].dtype}): NaN-freq [{nan_freq}%], Samples {sampled_values}\n"
    return _get_prompt(df, nl_prompt, how_many, samples=samples, pipeline_summary=pipeline_summary)


def _pipeline_summary_info(pipeline_summary):
    if pipeline_summary is not None:
        return (
            f" for a {pipeline_summary.task_type} task, predicting `{pipeline_summary.target_name}` "
            f"with {pipeline_summary.model}"
        )
    return ""


def _add_memorized_history(
    memory: list[dict[str, Any]] | None,
    messages: list[dict[str, str]],
    generated_code: list[str],
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
                generated_code.append(memorized_code)
                add_feature_sentence = "The code was executed and changes to ´df´ were kept."
                current_score = memorized_score
            else:
                add_feature_sentence = f"The last code changes to ´df´ were discarded. " f"(Improvement: {improvement})"

            messages += [
                {"role": "assistant", "content": memorized_code},
                {
                    "role": "user",
                    "content": f"Performance after adding feature: {target_metric}={memorized_score:.5f}. "
                    f".{add_feature_sentence}\nNext codeblock:\n",
                },
            ]


def _try_to_execute(df: pd.DataFrame, code_to_execute: str) -> tuple[list[str], list[str]]:
    df_sample = df.head(100).copy(deep=True)
    columns_before = df_sample.columns
    df_sample_processed = safe_exec(code_to_execute, "df", safe_locals_to_add={"df": df_sample})
    columns_after = df_sample_processed.columns
    new_columns = list(sorted(set(columns_after) - set(columns_before)))
    removed_columns = list(sorted(set(columns_before) - set(columns_after)))
    print(
        f"\t> Computed {len(new_columns)} new feature columns: {new_columns}, "
        f"removed {len(removed_columns)} feature columns: {removed_columns}"
    )
    return new_columns, removed_columns


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
    ) -> None:
        self.nl_prompt = nl_prompt
        self.how_many = how_many
        self.eval_mode = eval_mode
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory

        self.generated_code_: list[str] = []
        self.new_columns_: list[str] = []
        self.removed_columns_: list[str] = []

    def empty_state(self):
        return {"generated_code": []}

    def state_after_fit(self):
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self):
        if self.generated_code_ is not None and len(self.generated_code_) > 0:
            return self.generated_code_[-1]
        return OptimisableMixin.EMPTY_MEMORY_UPDATE

    def fit(self, df: pd.DataFrame, y=None, **fit_params):  # pylint: disable=unused-argument
        prompt_preview = self.nl_prompt[:40].replace("\n", " ").strip()

        if self._prefitted_state is not None:
            print(f"--- Using provided state for sempipes.with_sem_features('{prompt_preview}...', {self.how_many})")
            self.generated_code_ = self._prefitted_state["generated_code"]
            return self

        print(
            f"--- Fitting sempipes.with_sem_features('{prompt_preview}...', {self.how_many}) on dataframe of shape {df.shape} in mode '{self.eval_mode}'."
        )

        target_metric = "accuracy"
        if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
            target_metric = self._pipeline_summary.target_metric

        messages = []
        for attempt in range(1, _MAX_RETRIES + 1):
            code = ""

            try:  # pylint: disable=too-many-try-statements
                prompt = _build_prompt_from_df(df, self.nl_prompt, self.how_many, self._pipeline_summary)

                if attempt == 1:
                    messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                    _add_memorized_history(self._memory, messages, self.generated_code_, target_metric)

                code = generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(self.generated_code_)
                code_to_execute += "\n\n" + code

                new_columns, removed_columns = _try_to_execute(df, code_to_execute)

                self.generated_code_.append(code)
                self.new_columns_ = new_columns
                self.removed_columns_ = removed_columns

                break
            except Exception as e:  # pylint: disable=broad-except
                print(f"\t> An error occurred in attempt {attempt}:", e)
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        + f"Code: ```python{code}```\n Generate next feature (fixing error?):\n```python\n",
                    },
                ]

        return self

    def transform(self, df):
        check_is_fitted(self, ("generated_code_", "new_columns_", "removed_columns_"))
        code_to_execute = "\n".join(self.generated_code_)
        df = safe_exec(code_to_execute, "df", safe_locals_to_add={"df": df})

        for column in self.new_columns_:
            assert column in df.columns, f"Expected new column '{column}' not found in transformed dataframe"

        for column in self.removed_columns_:
            assert (
                column not in df.columns
            ), f"Expected removed column '{column}' still present in transformed dataframe"

        return df


class WithSemFeaturesCaafe(WithSemFeaturesOperator):
    def generate_features_estimator(self, data_op: DataOp, nl_prompt: str, name: str, how_many: int):
        _pipeline_summary = skrub.var(f"sempipes_pipeline_summary__{name}", None)
        _prefitted_state = skrub.var(f"sempipes_prefitted_state__{name}", None)
        _memory = skrub.var(f"sempipes_memory__{name}", [])

        return LLMFeatureGenerator(
            nl_prompt,
            how_many,
            _pipeline_summary=_pipeline_summary,
            _prefitted_state=_prefitted_state,
            _memory=_memory,
        )


def with_sem_features(
    self: DataOp,
    nl_prompt: str,
    name: str,
    how_many: int = 10,
) -> DataOp:
    data_op = self
    feature_gen_estimator = WithSemFeaturesCaafe().generate_features_estimator(data_op, nl_prompt, name, how_many)
    return self.skb.apply(feature_gen_estimator, how="no_wrap").skb.set_name(name)
