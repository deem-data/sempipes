from random import randint
from typing import Any, Self

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
    EstimatorTransformer,
    OptimisableMixin,
    SemCleanOperator,
)

_MAX_RETRIES = 5
_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates data cleaning code. Use skrub, sklearn, pandas, and scipy libraries as needed."
    "Answer only with the Python code."
)


def _try_to_execute_clean_code(df: pd.DataFrame, columns: list[str], code_to_execute: str) -> None:
    df_sample = df.head(50).copy(deep=True)

    clean_func = safe_exec(code_to_execute, "clean_columns")
    cleaned = clean_func(df_sample[columns], columns)

    assert isinstance(cleaned, pd.DataFrame), "clean_columns must return a pandas.DataFrame"
    assert cleaned.shape[0] == df_sample.shape[0], "cleaning must not change number of rows"
    print("\t> Code executed successfully on a sample dataframe for cleaning.")


def _build_df_sample(df: pd.DataFrame) -> str:
    samples = ""
    df_ = df.head(10)
    for column in list(df_):
        null_ratio = df[column].isna().mean()
        nan_freq = f"{null_ratio * 100:.2g}"
        sampled_values = df_[column].tolist()
        if str(df[column].dtype) == "float64":
            sampled_values = [round(sample, 2) for sample in sampled_values]
        samples += f"{df_[column].name} ({df[column].dtype}): NaN-freq [{nan_freq}%], Samples {sampled_values}\n"
    return samples


def _build_code_prompt(nl_prompt: str, df: pd.DataFrame, columns: list[str], data_description: str) -> str:
    prompt = f"""
    Given a pandas DataFrame `df` with columns {columns}, implement a function
    `clean_columns(df: pandas.DataFrame, columns: list[str]) -> pandas.DataFrame` that returns the dataframe with
    cleaned values for these columns.
    
    The data scientist wants you to take special care of the following: {nl_prompt}.

    Provide only a single Python method `def clean_columns(df, columns):` and returns a cleaned df.

    The preview of current column values is: {_build_df_sample(df)}.

    The descriptive statistics of the columns are: {data_description}.

    The cleaning should not change the number of rows; the cleaned data frame should contain {df.shape[0]} rows.
    """
    prompt += """
    Example:
    ```python
    import numpy as np
    import pandas as pd
    from typing import Iterable


    def _looks_like_date(series: pd.Series, sample_frac: float = 0.2, threshold: float = 0.6) -> bool:
        sample = series.dropna().sample(frac=min(sample_frac, 1.0)) if not series.dropna().empty else pd.Series([], dtype=object)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        success = parsed.notna().sum()
        return (success / len(sample)) >= threshold


    def _looks_like_numeric(series: pd.Series, sample_frac: float = 0.2, threshold: float = 0.6) -> bool:
        sample = series.dropna().sample(frac=min(sample_frac, 1.0)) if not series.dropna().empty else pd.Series([], dtype=object)
        if sample.empty:
            return False
        coerced = pd.to_numeric(sample.astype(str).str.strip().replace({"": None}), errors="coerce")
        success = coerced.notna().sum()
        return (success / len(sample)) >= threshold


    def clean_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        # Data-driven cleaning for specified columns.

        # - Infers whether each column should be treated as text, numeric, or date.
        # - For text-like columns: strip whitespace, lowercase, normalize common missing tokens.
        # - For numeric-like columns: coerce to numeric (errors->NaN), impute median if many missing after coercion,
        # cap outliers by IQR, keep dtype numeric.
        # - For date-like columns: parse to datetime (errors->NaT).
        # - Drops duplicate rows after cleaning.

        # Only arguments are `df` and `columns`. All other behavior is inferred from the data.
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        cols = [c for c in columns if c in df.columns]
        out = df.copy(deep=True)

        # Basic missing token normalization for strings
        missing_tokens = {"", "na", "n/a", "null", "none"}
        for c in cols:
            # Work on object/string-like columns first for missing token normalization
            if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object:
                out[c] = out[c].replace(lambda x: np.nan if isinstance(x, str) and x.strip().lower() in missing_tokens else x)

        # Decide per-column action
        numeric_cols = []
        date_cols = []
        text_cols = []
        for c in cols:
            ser = out[c]
            if _looks_like_date(ser):
                date_cols.append(c)
            elif _looks_like_numeric(ser):
                numeric_cols.append(c)
            else:
                text_cols.append(c)

        # Text cleaning: trim + lowercase
        for c in text_cols:
            out[c] = out[c].astype("string").str.strip()
            # only lowercase if many strings are mixed case
            sample = out[c].dropna().head(20).astype(str)
            if len(sample) > 0:
                # if any uppercase letters present in sample, lowercase for normalization
                any_upper = any(any(ch.isupper() for ch in val) for val in sample)
                if any_upper:
                    out[c] = out[c].str.lower()

        # Date parsing
        for c in date_cols:
            out[c] = pd.to_datetime(out[c], errors="coerce")

        # Numeric coercion
        for c in numeric_cols:
            out[c] = pd.to_numeric(out[c].astype(str).str.strip().replace({"": None}), errors="coerce")

        # Outlier capping (IQR) on numeric cols
        for c in numeric_cols:
            col = out[c].dropna()
            if col.empty:
                continue
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out[c] = out[c].clip(lower=lower, upper=upper)

        # Numeric imputation: if many NaNs after coercion, impute median
        for c in numeric_cols:
            num_missing = out[c].isna().sum()
            total = len(out[c])
            if total == 0:
                continue
            if num_missing / total > 0.1:  # heuristic: if >10% missing, impute
                median = out[c].median()
                out[c] = out[c].fillna(median)
        
        return out
        ```end

    """
    return prompt


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
                add_feature_sentence = (
                    f"The last cleaning code changes were discarded "
                    f"(improvement: {improvement}). Please make more improvements to the cleaning code."
                )

            messages += [
                {"role": "assistant", "content": memorized_code},
                {
                    "role": "user",
                    "content": f"Performance after cleaning selected columns with suggested code: {target_metric}={memorized_score:.5f}. "
                    f"{add_feature_sentence}\nNext codeblock:\n",
                },
            ]


# pylint: disable=too-many-ancestors
class LLMCleaner(BaseEstimator, TransformerMixin, ContextAwareMixin, OptimisableMixin):
    """Transformer that generates python cleaning code via the LLM.

    The transformer asks the LLM to produce a `clean_columns(df)` function and executes it.
    """

    def __init__(
        self,
        nl_prompt: str,
        columns: list[str],
        eval_mode: str = skrub.eval_mode(),
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
    ) -> None:
        self.nl_prompt = nl_prompt
        self.columns = columns
        self.eval_mode = eval_mode
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory
        self.generated_code_: list[str] = []

    def empty_state(self):
        return {"generated_code": []}

    def state_after_fit(self):
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self):
        if self.generated_code_ is not None and len(self.generated_code_) > 0:
            return self.generated_code_[-1]
        return OptimisableMixin.EMPTY_MEMORY_UPDATE

    def fit(self, df: pd.DataFrame, y=None) -> Self:  # pylint: disable=unused-argument
        print(f"--- sempipes.sem_clean(columns={self.columns}, nl_prompt='{self.nl_prompt}')")
        prompt_preview = self.nl_prompt[:40].replace("\n", " ").strip()

        if self._prefitted_state is not None:
            print(f"--- Using provided state for sempipes.sem_clean('{prompt_preview}...', {self.columns})")
            self.generated_code_ = self._prefitted_state["generated_code"]

            # If state was provided but generated_code is empty, we still need to generate code
            if not self.generated_code_:
                print("\t> Warning: Provided state has empty generated_code, generating new code...")
            else:
                return self

        target_metric = "accuracy"
        if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
            target_metric = self._pipeline_summary.target_metric

        data_description = str(df[self.columns].describe(include="all"))

        prompt = _build_code_prompt(
            df=df[self.columns], nl_prompt=self.nl_prompt, columns=self.columns, data_description=data_description
        )
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

        generated_code: list[str] = []
        for attempt in range(1, _MAX_RETRIES + 1):
            code = ""
            try:
                if attempt == 1:
                    _add_memorized_history(self._memory, messages, self.generated_code_, target_metric)

                code = generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(generated_code) + "\n\n" + code

                _try_to_execute_clean_code(df[self.columns], self.columns, code_to_execute)

                generated_code.append(code)
                self.generated_code_ = generated_code
                break
            except Exception as e:  # pylint: disable=broad-except
                print(f"\t> An error occurred in attempt {attempt}:", e)
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\nCode: ```python{code}```\nPlease generate corrected code:\n```python\n",
                    },
                ]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ("generated_code_",))

        print(self.generated_code_)

        code_to_execute = "\n".join(self.generated_code_)
        clean_func = safe_exec(code_to_execute, "clean_columns")
        df[self.columns] = clean_func(df[self.columns], self.columns)
        return df


class SemCleanWithLLM(SemCleanOperator):
    def generate_cleaning_estimator(
        self, data_op: DataOp, nl_prompt: str, columns: list[str], name: str
    ) -> EstimatorTransformer:
        _pipeline_summary = skrub.var(f"sempipes_pipeline_summary__{name}", None)
        _prefitted_state = skrub.var(f"sempipes_prefitted_state__{name}", None)
        _memory = skrub.var(f"sempipes_memory__{name}", [])
        return LLMCleaner(
            nl_prompt=nl_prompt,
            columns=columns,
            eval_mode=skrub.eval_mode(),
            _pipeline_summary=_pipeline_summary,
            _prefitted_state=_prefitted_state,
            _memory=_memory,
        )


def sem_clean(self: DataOp, nl_prompt: str, columns: list[str], name: str | None = None) -> DataOp:
    if name is None:
        name = f"random_{randint(0, 10000)}_name"

    data_op = self
    cleaner = SemCleanWithLLM().generate_cleaning_estimator(
        data_op=data_op, nl_prompt=nl_prompt, columns=columns, name=name
    )
    return self.skb.apply(cleaner, how="no_wrap").skb.set_name(name)
