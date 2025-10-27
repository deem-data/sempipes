import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.operators.operators import SemDistillDataOperator

_SYSTEM_PROMPT = """
You are an expert data scientist, assisting with data distillation and condensation. You answer by generating code.
"""
_MAX_RETRIES = 5


def _get_samples_from_df(df: pd.DataFrame, number_of_samples: int = 10) -> str:
    samples = ""
    df_ = df.sample(min(number_of_samples, len(df)))
    for column in list(df_):
        null_ratio = df[column].isna().mean()
        nan_freq = f"{null_ratio * 100:.2g}"
        sampled_values = df_[column].tolist()
        if str(df[column].dtype) == "float64":
            sampled_values = [round(sample, 2) for sample in sampled_values]
        samples += f"{df_[column].name} ({df[column].dtype}): NaN-freq [{nan_freq}%], Samples {sampled_values}\n"
    return samples


def _try_to_execute(df: pd.DataFrame, code_to_execute: str) -> None:
    df_sample = df.head(100).copy(deep=True)
    number_of_rows = 50
    columns_before = df_sample.columns

    distillation_func = safe_exec(code_to_execute, "distill_data")
    df_sample_processed = distillation_func(df_sample, number_of_rows)
    columns_after = df_sample_processed.columns

    if sorted(set(columns_before)) != sorted(set(columns_after)):
        raise ValueError("\t> Code execution changed columns.")
    if df_sample_processed.shape[0] != number_of_rows:
        raise ValueError(
            f"\t> Code execution generated a wrong number of rows: {df_sample_processed.shape[0]} instead of the expected {number_of_rows} rows. Return a variable distilled in-place and named `df`."
        )

    print(f" Generated {df_sample_processed.shape[0]} rows from a pd.DataFrame with {df_sample.shape[0]} rows.")


class CodeDataDistiller(BaseEstimator, TransformerMixin):
    def __init__(self, nl_prompt: str, number_of_rows: int):
        self.nl_prompt = nl_prompt
        self.number_of_rows = number_of_rows
        self.generated_code_: list[str] = []

    @staticmethod
    def _build_prompt_for_code_generation(nl_prompt, data_description_unparsed, samples, df):
        return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes. Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

You need to generate Python code for the in-place data distillation of the dataframe `df` that returns a distilled and smaller version of the dataframe with `number_of_rows` rows. 
The generated code should be a Python method `distill_data(df: pandas.DataFrame, number_of_rows: int) -> pandas.DataFrame` that takes as input a pandas DataFrame with original data and an integer with number of rows to distill, and returns a distilled DataFrame `df_distilled` with `number_of_rows` rows. 

Number of samples (rows) in training dataset: {int(len(df))}.
Number of samples (rows) to distill: `number_of_rows`.

The data scientist wants you to take special care to the following: {nl_prompt}.

You are allowed to use `torch`, `torchvision`, `transformers` libraries. Additionally, you can leverage methods as subsampling, SMOTE, gradient matching, gradient trajectory matching, feature space matching, generative models such as GANs or diffusion, coreset selection, and other advanced techniques.

Code formatting:
```python
def distill_data(df: pandas.DataFrame, number_of_rows: int) -> pandas.DataFrame:
    # Some Python code to distill df into df_distilled and return a distilled df_distilled
    ...
    assert df_distilled.shape[0] == number_of_rows, f"Output df must have {{number_of_rows}} rows, but distilled to {{df.shape[0]}} rows."
    return df_distilled
```end

Each codeblock ends with ```end and starts with "```python"

Return only Python code, no explanations.

Codeblock:
"""

    def fit_transform(self, X, y=None, **kwargs):  # pylint: disable=unused-argument
        print(f"--- sempipes.sem_distill('{self.nl_prompt}', {self.number_of_rows})")
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
                )
                if attempt == 1:
                    messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                code = generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(self.generated_code_)
                code_to_execute += "\n\n" + code

                # Try to execute the code
                _try_to_execute(X, code_to_execute)

                self.generated_code_.append(code)
                break
            except Exception as e:  # pylint: disable=broad-except
                print(f"\t> An error occurred in attempt {attempt}:", e)
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        f"Code: ```python{code}```\n Generate code again (fixing error?):\n```python\n",
                    },
                ]
        code_to_execute = "\n".join(self.generated_code_)
        distill_func = safe_exec(code_to_execute, "distill_data")
        df_distilled = distill_func(X, self.number_of_rows)

        print(f"\t> Generated code: {code_to_execute}")

        return df_distilled

    def transform(self, df):
        return df


class SemDistillData(SemDistillDataOperator):
    def generate_data_distiller(self, nl_prompt: str, number_of_rows: int):
        return CodeDataDistiller(nl_prompt=nl_prompt, number_of_rows=number_of_rows)


def sem_distill(self: DataOp, nl_prompt: str, number_of_rows: int) -> DataOp:
    data_distiller = SemDistillData().generate_data_distiller(nl_prompt=nl_prompt, number_of_rows=number_of_rows)
    return self.skb.apply(data_distiller, how="no_wrap")
