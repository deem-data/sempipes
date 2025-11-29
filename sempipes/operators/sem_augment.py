from io import StringIO

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.llm.llm import batch_generate_json_retries, generate_python_code_from_messages
from sempipes.logging import get_logger
from sempipes.operators.operators import SemAugmentDataOperator

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
    def generate_data_generator(self, nl_prompt: str, number_of_rows_to_generate: int, **kwargs):
        generate_via_code = kwargs["generate_via_code"] if "generate_via_code" in kwargs else True
        if generate_via_code:
            return CodeDataAugmentor(nl_prompt=nl_prompt, number_of_rows_to_generate=number_of_rows_to_generate)
        return DirectDataAugmentor(nl_prompt=nl_prompt, number_of_rows_to_generate=number_of_rows_to_generate)


def _try_to_execute(df: pd.DataFrame, code_to_execute: str, number_of_rows_to_generate: int) -> None:
    df_sample = df.head(100).copy(deep=True)
    columns_before = df_sample.columns

    logger.info("Validation generated code...")
    augmentation_func = safe_exec(code_to_execute, "augment_data")
    df_sample_processed = augmentation_func(df_sample)

    columns_after = df_sample_processed.columns
    column_difference = set(columns_before) - set(columns_after)

    if sorted(set(columns_before)) != sorted(set(columns_after)):
        raise ValueError(f"\t> Code execution changed columns: {column_difference}")
    if df_sample_processed.shape[0] != number_of_rows_to_generate + df_sample.shape[0]:
        raise ValueError(
            f"\t> Code execution generated a wrong number of rows: {df_sample_processed.shape[0] - 100} instead of the expected {number_of_rows_to_generate} rows. Return a variable augmented in-place and named `df`. Check that new rows are actually generated and appended to the original df."
        )

    print(f" Generated {df_sample_processed.shape[0]} rows from a pd.DataFrame with {df_sample.shape[0]} rows.")


class CodeDataAugmentor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        nl_prompt: str,
        number_of_rows_to_generate: int,
    ) -> None:
        self.nl_prompt = nl_prompt
        self.number_of_rows_to_generate = number_of_rows_to_generate
        self.generated_code_: list[str] = []

    @staticmethod
    def _build_prompt_for_code_generation(
        nl_prompt, data_description_unparsed, samples, number_of_rows_to_generate, df
    ):
        return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

Number of samples (rows) in training dataset: {int(len(df))}.
Number of samples (rows) to augment: {int(number_of_rows_to_generate)}.

You need to generate Python code for the in-place data augmentation of the dataframe `df` that returns the original dataframe `df` with appended augmented rows. 
The generated code should be a Python method `augment_data(df: pandas.DataFrame) -> pandas.DataFrame` that takes as input a pandas DataFrame and returns the same pandas DataFrame `df` with appended {number_of_rows_to_generate} new augmented rows`. 

The data scientist wants you to take special care of the following: {nl_prompt}.

Here is a simple example:

```python
def augment_data(df: pandas.DataFrame) -> pandas.DataFrame:
    from sdv.metadata import Metadata
    from sdv.single_table import TVAESynthesizer

    num_rows_to_synth = 10

    metadata = Metadata.detect_from_dataframe(data=df, table_name='train_data')
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(data=df)
    augmented_data = synthesizer.sample(num_rows=num_rows_to_synth)
    df = pd.concat([df, augmented_data], ignore_index=True)
    return df
```end

Here is a more complex example, where we condition the synthetic data generation on a specific column value:

```python
def augment_data(df: pandas.DataFrame) -> pandas.DataFrame:
    from sdv.metadata import Metadata
    from sdv.single_table import TVAESynthesizer
    from sdv.sampling import Condition

    num_rows_to_synth = 100

    metadata = Metadata.detect_from_dataframe(data=df, table_name='train_data')
    synthesizer = TVAESynthesizer(metadata)
    conditioned = Condition(num_rows=num_rows_to_synth, column_values={{'column_to_condition_on': 'column_value_to_condition_on'}})
    synthetic_train = synthesizer.sample_from_conditions([conditioned])
    df = pd.concat([df, synthetic_train], ignore_index=True)
    return df
```end

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
        logger.info(f"sempipes.sem_augment('{self.nl_prompt}', True, {self.number_of_rows_to_generate})")

        messages = []
        for attempt in range(1, _MAX_RETRIES + 1):
            code = ""

            try:
                samples = _get_samples_from_df(X, number_of_samples=10)
                prompt = self._build_prompt_for_code_generation(
                    df=X,
                    nl_prompt=self.nl_prompt,
                    data_description_unparsed=X.describe(include="all").to_string(),
                    samples=samples,
                    number_of_rows_to_generate=self.number_of_rows_to_generate,
                )

                if attempt == 1:
                    messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

                code = generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(self.generated_code_)
                code_to_execute += "\n\n" + code

                # print("#" * 80)
                # print(f"{code}")
                # print("#" * 80)

                _try_to_execute(X.copy(deep=True), code_to_execute, self.number_of_rows_to_generate)

                self.generated_code_.append(code)
                break
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f"\t> An error occurred in attempt {attempt}: {e}", exc_info=True)
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        f"Code: ```python{code}```\n Generate code again (fixing error?):\n```python\n",
                    },
                ]

        code_to_execute = "\n".join(self.generated_code_)
        augmentation_func = safe_exec(code_to_execute, "augment_data")
        df_augmented = augmentation_func(X.copy(deep=True))
        return df_augmented

    def transform(self, df):
        return df


class DirectDataAugmentor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        nl_prompt: str,
        number_of_rows_to_generate: int,
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
    **kwargs,
) -> DataOp:
    data_augmentor = SemAugmentData().generate_data_generator(
        nl_prompt=nl_prompt, number_of_rows_to_generate=number_of_rows_to_generate, **kwargs
    )

    return self.skb.apply(data_augmentor, how="no_wrap")
