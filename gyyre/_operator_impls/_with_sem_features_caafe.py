# TODO This class needs some serious cleanup / refactoring
from typing import Any

import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from gyyre._code_gen._exec import _safe_exec
from gyyre._code_gen._llm import _generate_python_code_from_messages
from gyyre._operators import (
    GyyreContextAwareMixin,
    GyyreOptimisableMixin,
    GyyrePrefittableMixin,
    WithSemFeaturesOperator,
)
from gyyre.optimisers._dag_summary import DagSummary

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
    dag_summary: DagSummary | None = None,
) -> str:
    data_description_unparsed = None

    task_description = (
        "This code generates additional columns that are useful for a downstream classification "
        "algorithm (such as XGBoost) predicting a target label."
    )
    usefulness = ""
    model_reference = "classifier"

    if dag_summary is not None:
        task_type = dag_summary.task_type
        model = dag_summary.model
        target_name = dag_summary.target_name
        data_description_unparsed = dag_summary.dataset_description

        if task_type and model and target_name:
            task_description = (
                f"This code generates additional columns that are useful for a "
                f'downstream {task_type} algorithm ({model}) predicting "{target_name}".'
            )

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
evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks.

The data scientist wants you to take special care to the following: {nl_prompt}.


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
    df: pd.DataFrame, nl_prompt: str, how_many: int, dag_summary: DagSummary | None = None
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
    return _get_prompt(df, nl_prompt, how_many, samples=samples, dag_summary=dag_summary)


def _dag_summary_info(gyyre_dag_summary):
    if gyyre_dag_summary is not None:
        return (
            f" for a {gyyre_dag_summary.task_type} task, predicting `{gyyre_dag_summary.target_name}` "
            f"with {gyyre_dag_summary.model}"
        )
    return ""


def _add_memorized_history(
    gyyre_memory: list[dict[str, Any]] | None,
    messages: list[dict[str, str]],
    generated_code: list[str],
) -> None:
    if gyyre_memory is not None and len(gyyre_memory) > 0:
        current_accuracy = 0.0
        current_roc = 0.0

        for memory_line in gyyre_memory:
            memorized_code = memory_line["update"]
            memorized_accuracy = memory_line["score"]
            # TODO also compute and provide ROC AUC
            memorized_roc = memory_line["score"]

            improvement_acc = memorized_accuracy - current_accuracy
            improvement_roc = memorized_roc - current_roc

            if improvement_roc + improvement_acc >= 0.0:
                generated_code.append(memorized_code)
                add_feature_sentence = "The code was executed and changes to ´df´ were kept."
                current_accuracy = memorized_accuracy
                current_roc = memorized_roc
            else:
                add_feature_sentence = (
                    f"The last code changes to ´df´ were discarded. "
                    f"(Improvement: {improvement_roc + improvement_acc})"
                )

            messages += [
                {"role": "assistant", "content": memorized_code},
                {
                    "role": "user",
                    "content": f"Performance after adding feature ROC {memorized_roc:.3f}, "
                    f"ACC {memorized_accuracy:.3f}. {add_feature_sentence}\nNext codeblock:\n",
                },
            ]


def _try_to_execute(df: pd.DataFrame, code_to_execute: str) -> None:
    df_sample = df.head(100).copy(deep=True)
    columns_before = df_sample.columns
    # print("-" * 120)
    # print(code_to_execute)
    # print("-" * 120)
    df_sample_processed = _safe_exec(code_to_execute, "df", safe_locals_to_add={"df": df_sample})
    columns_after = df_sample_processed.columns
    new_columns = sorted(set(columns_after) - set(columns_before))
    removed_columns = sorted(set(columns_before) - set(columns_after))
    print(
        f"\t> Computed {len(new_columns)} new feature columns: {new_columns}, "
        f"removed {len(removed_columns)} feature columns: {removed_columns}"
    )


# pylint: disable=too-many-ancestors
class LLMFeatureGenerator(
    BaseEstimator, TransformerMixin, GyyreContextAwareMixin, GyyrePrefittableMixin, GyyreOptimisableMixin
):
    def __init__(
        self,
        nl_prompt: str,
        how_many: int,
        gyyre_dag_summary: DagSummary | None | DataOp = None,
        gyyre_prefitted_state: dict[str, Any] | None | DataOp = None,
        gyyre_memory: list[dict[str, Any]] | None | DataOp = None,
    ) -> None:
        self.nl_prompt = nl_prompt
        self.how_many = how_many
        self.gyyre_dag_summary = gyyre_dag_summary
        self.gyyre_prefitted_state = gyyre_prefitted_state
        self.gyyre_memory = gyyre_memory

        self.generated_code_ = []

    def state_after_fit(self):
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self):
        if self.generated_code_ is not None and len(self.generated_code_) > 0:
            return self.generated_code_[-1]
        return ""

    def fit(self, df: pd.DataFrame, y=None, **fit_params):  # pylint: disable=unused-argument
        prompt_preview = self.nl_prompt[:40].replace("\n", " ").strip()

        if self.gyyre_prefitted_state is not None:
            print(f"--- Using provided state for gyyre.with_sem_features('{prompt_preview}...', {self.how_many})")
            self.generated_code_ = self.gyyre_prefitted_state["generated_code"]
            return self

        print(
            f"--- Fitting gyyre.with_sem_features('{prompt_preview}...', {self.how_many})"
            f"{_dag_summary_info(self.gyyre_dag_summary)}"
        )

        messages = []
        for attempt in range(1, _MAX_RETRIES + 1):
            code = ""

            try:
                prompt = _build_prompt_from_df(df, self.nl_prompt, self.how_many, self.gyyre_dag_summary)

                if attempt == 1:
                    messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                    _add_memorized_history(self.gyyre_memory, messages, self.generated_code_)

                code = _generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(self.generated_code_)
                code_to_execute += "\n\n" + code

                _try_to_execute(df, code_to_execute)

                self.generated_code_.append(code)
                break
            except Exception as e:  # pylint: disable=broad-except
                print(f"\t> An error occurred in attempt {attempt}:", e)
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        f"Code: ```python{code}```\n Generate next feature (fixing error?):\n```python\n",
                    },
                ]

        return self

    def transform(self, df):
        check_is_fitted(self, "generated_code_")
        code_to_execute = "\n".join(self.generated_code_)
        df = _safe_exec(code_to_execute, "df", safe_locals_to_add={"df": df})
        return df


class WithSemFeaturesCaafe(WithSemFeaturesOperator):
    def generate_features_estimator(self, data_op: DataOp, nl_prompt: str, name: str, how_many: int):
        gyyre_dag_summary = skrub.var(f"gyyre_dag_summary__{name}", None)
        gyyre_prefitted_state = skrub.var(f"gyyre_prefitted_state__{name}", None)
        gyyre_memory = skrub.var(f"gyyre_memory__{name}", [])

        return LLMFeatureGenerator(
            nl_prompt,
            how_many,
            gyyre_dag_summary=gyyre_dag_summary,
            gyyre_prefitted_state=gyyre_prefitted_state,
            gyyre_memory=gyyre_memory,
        )
