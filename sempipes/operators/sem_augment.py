from io import StringIO
from typing import Any

import pandas as pd
import sdv
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.llm.llm import batch_generate_json_retries, generate_python_code_from_messages
from sempipes.logging import get_logger
from sempipes.operators.operators import ContextAwareMixin, OptimisableMixin, SemAugmentDataOperator

logger = get_logger()

_SYSTEM_PROMPT = """
You are an expert data scientist, assisting with data augmentation. You answer by generating code or directly generating data.
"""

_MAX_RETRIES = 10


def _get_samples_from_df(
    df: pd.DataFrame,
    number_of_samples: int = 10,
) -> str:
    samples = ""
    df_ = df.sample(number_of_samples)
    for column in list(df_):
        null_ratio = df[column].isna().mean()
        nan_freq = f"{null_ratio * 100:.2g}"
        sampled_values = df_[column].tolist()
        if str(df[column].dtype) == "float64":
            sampled_values = [round(sample, 2) for sample in sampled_values]
        samples += f"{df_[column].name} ({df[column].dtype}): NaN-freq [{nan_freq}%], Samples {sampled_values}\n"
    return samples


class SemAugmentData(SemAugmentDataOperator):
    def generate_data_generator(
        self,
        data_op: DataOp,
        nl_prompt: str,
        name: str,
        number_of_rows_to_generate: int,
        eval_mode: str = skrub.eval_mode(),
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
        **kwargs,
    ):
        generate_via_code = kwargs["generate_via_code"] if "generate_via_code" in kwargs else True
        if generate_via_code:
            return CodeDataAugmentor(
                nl_prompt=nl_prompt,
                number_of_rows_to_generate=number_of_rows_to_generate,
                eval_mode=eval_mode,
                _pipeline_summary=_pipeline_summary,
                _prefitted_state=_prefitted_state,
                _memory=_memory,
                _inspirations=_inspirations,
            )
        return DirectDataAugmentor(nl_prompt=nl_prompt, number_of_rows_to_generate=number_of_rows_to_generate)


def _try_to_execute(df: pd.DataFrame, code_to_execute: str, number_of_rows_to_generate: int) -> None:
    df_sample = df.head(1000).copy(deep=True)
    columns_before = df_sample.columns

    logger.info("Validating generated code...")
    augmentation_func = safe_exec(code_to_execute, "augment_data")
    df_sample_processed = augmentation_func(df_sample)

    columns_after = df_sample_processed.columns
    column_difference = set(columns_before) - set(columns_after)

    if sorted(set(columns_before)) != sorted(set(columns_after)):
        raise ValueError(f"\t> Code execution changed columns: {column_difference}")
    if df_sample_processed.shape[0] != number_of_rows_to_generate + df_sample.shape[0]:
        raise ValueError(
            f"The code returned wrong number of rows: {df_sample_processed.shape[0] - 100} instead of the expected {number_of_rows_to_generate} rows."
        )

    logger.debug(f"Generated {df_sample_processed.shape[0]} rows from a pd.DataFrame with {df_sample.shape[0]} rows.")


def _add_memorized_history(
    memory: list[dict[str, Any]] | None,
    messages: list[dict[str, str]],
    target_metric: str,
) -> None:
    if memory is not None and len(memory) > 0:
        current_score = None

        messages += [
            {
                "role": "user",
                "content": """
                IMPORTANT: Try to generate code that improves the performance of the model. Here are some things that you can try:

                    - Increase or decrease the number of rows to synthesize
                    - Change the hyperparameters of the synthesizer, e.g. the number of epochs (if it supports it)
                    - Try different synthesizers, e.g. the GaussianCopulaSynthesizer or the TVAESynthesizer
                    - If you do conditional sampling, you can try to fit the synthesizer on a subset of the data that matches the condition

                Below is a history of the code that has been generated and executed so far, together with the performance of the model.

                """,
            }
        ]

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


class CodeDataAugmentor(BaseEstimator, TransformerMixin, ContextAwareMixin, OptimisableMixin):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        nl_prompt: str,
        number_of_rows_to_generate: int,
        eval_mode: str = skrub.eval_mode(),
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
    ) -> None:
        self.nl_prompt = nl_prompt
        self.number_of_rows_to_generate = number_of_rows_to_generate
        self.eval_mode = eval_mode
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory
        self._inspirations: list[dict[str, Any]] | DataOp | None = _inspirations
        self.generated_code_: str | None = None

    def empty_state(self):
        return {
            "generated_code": """
def augment_data(df: pandas.DataFrame) -> pandas.DataFrame:
    return df        
"""
        }

    def state_after_fit(self):
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self):
        if self.generated_code_ is not None:
            return self.generated_code_
        return OptimisableMixin.EMPTY_MEMORY_UPDATE

    @staticmethod
    def _build_prompt_for_code_generation(
        nl_prompt, samples, number_of_rows_to_generate, df, pipeline_summary, inspirations
    ):
        inspiration_examples = ""
        if inspirations and len(inspirations) > 0:
            inspiration_examples += (
                "Here are some examples of code that has been used to augment the data in the past, together with the resulting scores."
                "You can use these as inspiration to generate your own code.\n\n"
            )

            # Sort inspirations by score in descending order and select top 3
            top_inspirations = sorted(inspirations, key=lambda x: x["score"], reverse=True)[:3]

            for i, inspiration in enumerate(top_inspirations):
                code = "\n".join(inspiration["state"]["generated_code"])
                score = inspiration["score"]
                inspiration_examples += f"Example {i+1}:\n```python\n{code}\n```\nScore: {score:.4f}\n\n"

        data_description_from_pipeline = ""
        if pipeline_summary is not None:
            data_description_from_pipeline = (
                f"Here is how the data scientists describes the data contained in the dataframe `df`:"
                f"\n{pipeline_summary.dataset_description}"
            )

        return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.

{data_description_from_pipeline}

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

Number of samples (rows) in training dataset: {int(len(df))}.
Number of samples (rows) to augment: {int(number_of_rows_to_generate)}.

You need to generate Python code for the data augmentation of the dataframe `df` that returns the original dataframe `df` with appended augmented rows. 
The generated code should be a Python method `augment_data(df: pandas.DataFrame) -> pandas.DataFrame` that takes as input a pandas DataFrame and returns the same pandas DataFrame `df` with appended {number_of_rows_to_generate} new augmented rows`. 

The data scientist wants you to take special care of the following: {nl_prompt}.

Here is a simple example:

```python
def augment_data(df: pandas.DataFrame) -> pandas.DataFrame:
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    num_rows_to_synth = 10

    metadata = Metadata.detect_from_dataframe(data=df, table_name='train_data')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data=df)
    augmented_data = synthesizer.sample(num_rows=num_rows_to_synth)
    df = pd.concat([df, augmented_data], ignore_index=True)
    return df
```end

Here is a more complex example, where we condition the synthetic data generation on a specific column value:

```python
def augment_data(df: pandas.DataFrame) -> pandas.DataFrame:
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.sampling import Condition

    num_rows_to_synth = 100

    metadata = Metadata.detect_from_dataframe(data=df, table_name='train_data')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    # Fit the synthesizer on the original data
    synthesizer.fit(data=df)
    # Condition the synthetic data generation on a specific column value
    conditioned = Condition(num_rows=num_rows_to_synth, column_values={{'column_to_condition_on': 'column_value_to_condition_on'}})
    # Generate synthetic data conditioned on the specific column value
    synthetic_train = synthesizer.sample_from_conditions([conditioned])
    # Append the synthetic data to the original dataframe
    df = pd.concat([df, synthetic_train], ignore_index=True)
    return df
```end

{inspiration_examples}

The available version of the sdv library is: {sdv.__version__}.
Here is the exact code formatting for the function to generate:
```python
def augment_data(df: pandas.DataFrame) -> pandas.DataFrame:
    # Some Python code to augment df with new rows, append it to the original df, and return original df with new rows
    ...
    return df
```end

Each codeblock ends with ```end and starts with "```python"

Return only Python code, no explanations or apologies. Explain the code you generate with comments.

Codeblock:
"""

    def fit_transform(self, X, y=None, **kwargs):  # pylint: disable=unused-argument
        if self._prefitted_state is not None:
            self.generated_code_ = self._prefitted_state["generated_code"]
        else:
            logger.info(f"sempipes.sem_augment('{self.nl_prompt}', True, {self.number_of_rows_to_generate})")

            target_metric = "accuracy"
            if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
                target_metric = self._pipeline_summary.target_metric

            messages = []
            for attempt in range(1, _MAX_RETRIES + 1):
                code = ""

                try:  # pylint: disable=too-many-try-statements
                    samples = _get_samples_from_df(X, number_of_samples=10)
                    prompt = self._build_prompt_for_code_generation(
                        df=X,
                        nl_prompt=self.nl_prompt,
                        samples=samples,
                        number_of_rows_to_generate=self.number_of_rows_to_generate,
                        pipeline_summary=self._pipeline_summary,
                        inspirations=self._inspirations,
                    )

                    if attempt == 1:
                        messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                        _add_memorized_history(self._memory, messages, target_metric)

                    code = generate_python_code_from_messages(messages)

                    print("#" * 80)
                    print(f"{code}")
                    print("#" * 80)

                    _try_to_execute(X.copy(deep=True), code, self.number_of_rows_to_generate)

                    self.generated_code_ = code
                    break
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(f"\t> An error occurred in attempt {attempt}: {e}")  # , exc_info=True)
                    messages += [
                        {"role": "assistant", "content": code},
                        {
                            "role": "user",
                            "content": f"Code execution failed with error: {type(e)} {e}.\n "
                            f"Code: ```python{code}```\n Generate code again (fixing error?):\n```python\n",
                        },
                    ]

        print(">" * 80)
        print(f"{self.generated_code_}")
        print(">" * 80)

        if self.generated_code_ is None:
            logger.warning("No working code generated after {_MAX_RETRIES} retries. Falling back to empty state.")
            self.generated_code_ = self.empty_state()["generated_code"]

        augmentation_func = safe_exec(self.generated_code_, "augment_data")
        df_augmented = augmentation_func(X.copy(deep=True))
        return df_augmented

    def transform(self, df):
        return df


class DirectDataAugmentor(BaseEstimator, TransformerMixin, ContextAwareMixin, OptimisableMixin):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        nl_prompt: str,
        number_of_rows_to_generate: int,
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
    ) -> None:
        self.nl_prompt = nl_prompt
        self.number_of_rows_to_generate = number_of_rows_to_generate
        self.generated_json_: list[str] = []

    @staticmethod
    def _build_prompt_for_data_generation(
        nl_prompt, data_description_unparsed, samples, number_of_rows_to_generate, df
    ):
        prompt = f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

Number of samples (rows) in training dataset: {int(len(df))}.
Number of samples (rows) to augment: {int(number_of_rows_to_generate)}.

You need to augment dataframe `df` by generating {int(number_of_rows_to_generate)} new rows that are similar to the existing data, but with variations and augmentations that make sense given the context of the data.
Please provide the augmented data in a JSON array format, where each element in the array represents a new row to be added to the dataframe. 
Each row should be a JSON object with key-value pairs corresponding to column names and their respective values.
Please ensure that the data types of the values match the data types of the existing columns in the dataframe.
The generated JSON array should be readable via `pandas.read_json(StringIO(_), orient='records')`.

Generate data only for the existing columns: {df.columns.to_list()}.


The data scientist wants you to take special care of the following: {nl_prompt}.
"""

        prompt += """
Code formatting for each added column:
```json
[{"col 1":"a","col 2":"b"},{"col 1":"c","col 2":"d"}]
```end

Each codeblock ends with ```end and starts with "```json"
Codeblock:
"""
        return prompt

    def empty_state(self):
        raise NotImplementedError()

    def state_after_fit(self):
        raise NotImplementedError()

    def memory_update_from_latest_fit(self):
        raise NotImplementedError()

    def fit_transform(self, X, y=None, **kwargs):  # pylint: disable=unused-argument
        logger.info(f"sempipes.sem_augment('{self.nl_prompt}', False, {self.number_of_rows_to_generate})")

        all_messages = []
        batch_size = 100
        total = self.number_of_rows_to_generate
        for start in range(0, total, batch_size):
            current_batch_size = min(batch_size, total - start)
            samples = _get_samples_from_df(X, number_of_samples=10)
            prompt = self._build_prompt_for_data_generation(
                df=X,
                nl_prompt=self.nl_prompt,
                data_description_unparsed=X.describe(include="all").to_string(),
                samples=samples,
                number_of_rows_to_generate=current_batch_size,
            )

            all_messages.append([{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}])

        all_augmented_data = batch_generate_json_retries(all_messages)

        all_augmented_batches = []
        for augmented_data in all_augmented_data:
            self.generated_json_.append(augmented_data)
            df_augmented_batch = pd.read_json(StringIO(augmented_data), orient="records")
            all_augmented_batches.append(df_augmented_batch)

        self.df_augmented_ = pd.concat(all_augmented_batches, ignore_index=True)

        # Slice if too many rows were generated
        if self.df_augmented_.shape[0] > self.number_of_rows_to_generate:
            self.df_augmented_ = self.df_augmented_.sample(n=self.number_of_rows_to_generate, random_state=0)

        self.df_augmented_ = pd.concat([X, self.df_augmented_], ignore_index=True)

        return self.df_augmented_

    def transform(self, df):
        return df


def sem_augment(
    self: DataOp,
    nl_prompt: str,
    number_of_rows_to_generate: int,
    name: str,
    **kwargs,
) -> DataOp:
    _pipeline_summary = skrub.var(f"sempipes_pipeline_summary__{name}", None)
    _prefitted_state = skrub.var(f"sempipes_prefitted_state__{name}", None)
    _memory = skrub.var(f"sempipes_memory__{name}", [])
    _inspirations = skrub.var(f"sempipes_inspirations__{name}", [])

    data_augmentor = SemAugmentData().generate_data_generator(
        data_op=self,
        nl_prompt=nl_prompt,
        name=name,
        number_of_rows_to_generate=number_of_rows_to_generate,
        eval_mode=skrub.eval_mode(),
        _pipeline_summary=_pipeline_summary,
        _prefitted_state=_prefitted_state,
        _memory=_memory,
        _inspirations=_inspirations,
        **kwargs,
    )

    result = self.skb.apply(data_augmentor, how="no_wrap")

    # Workaround to make the fitted estimator available in the computational graph
    fitted_estimator = result.skb.applied_estimator.skb.set_name(f"sempipes_fitted_estimator__{name}")
    result_with_name = result.skb.set_name(name)
    result_with_fitted_estimator = skrub.as_data_op({"fitted_estimator": fitted_estimator, "result": result_with_name})

    def extract_result(tuple_of_data_ops):
        return tuple_of_data_ops["result"]

    return result_with_fitted_estimator.skb.apply_func(extract_result)
