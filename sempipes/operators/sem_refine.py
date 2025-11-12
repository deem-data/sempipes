import json
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
    SemRefineOperator,
)


class SemRefineWithLLM(SemRefineOperator):
    def generate_refinement_estimator(
        self,
        data_op: DataOp,
        target_column: str,
        nl_prompt: str,
        refine_with_existing_values_only: bool,
        name: str,
    ) -> EstimatorTransformer:
        _pipeline_summary = skrub.var(f"sempipes_pipeline_summary__{name}", None)
        _prefitted_state = skrub.var(f"sempipes_prefitted_state__{name}", None)
        _memory = skrub.var(f"sempipes_memory__{name}", [])
        return LLMDeduplicator(
            target_column=target_column,
            nl_prompt=nl_prompt,
            refine_with_existing_values_only=refine_with_existing_values_only,
            eval_mode=skrub.eval_mode(),
            _pipeline_summary=_pipeline_summary,
            _prefitted_state=_prefitted_state,
            _memory=_memory,
        )


_MAX_RETRIES = 5
_SYSTEM_PROMPT = (
    "You are a helpful assistant, assisting data scientists with improving their data preparation code and, for instance, refinement code. "
    "You answer only by generating code. Answer as concisely as possible."
)


def _try_to_execute(df: pd.DataFrame, target_column: str, code_to_execute: str) -> None:
    df_sample = df.head(50).copy(deep=True)

    refinement_func = safe_exec(code_to_execute, "refine_column")
    refined_df = refinement_func(df_sample)

    assert isinstance(refined_df, pd.DataFrame), "refine_column must return a pandas.DataFrame"
    assert (
        refined_df[target_column].isna().sum() == df_sample[target_column].isna().sum()
    ), "The refined column contains more NaN values."
    assert refined_df.shape[0] == df_sample.shape[0], "The refined column shape was modified."
    print("\t> Code executed successfully on a sample dataframe for refinement.")


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
                    f"The last refinement code changes were discarded "
                    f"(improvement: {improvement}). Please make more improvements to the refinement code and dictionaries if used. Consider changing or generalizing correspondences if possible."
                )

            messages += [
                {"role": "assistant", "content": memorized_code},
                {
                    "role": "user",
                    "content": f"Performance after refining selected column with suggested code: {target_metric}={memorized_score:.5f}. "
                    f"{add_feature_sentence}\nNext codeblock:\n",
                },
            ]


# pylint: disable=too-many-ancestors
class LLMDeduplicator(BaseEstimator, TransformerMixin, ContextAwareMixin, OptimisableMixin):
    """Transformer that generates python refinement code via the LLM."""

    def __init__(
        self,
        target_column: str,
        nl_prompt: str,
        refine_with_existing_values_only: bool,
        eval_mode: str = skrub.eval_mode(),
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
    ) -> None:
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.refine_with_existing_values_only = refine_with_existing_values_only
        self.eval_mode = eval_mode
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory
        self.generated_code_: list[str] = []
        self.value_counts_: str | None = None

    def empty_state(self):
        return {"generated_code": []}

    def state_after_fit(self):
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self):
        if self.generated_code_ is not None and len(self.generated_code_) > 0:
            return self.generated_code_[-1]
        return OptimisableMixin.EMPTY_MEMORY_UPDATE

    @staticmethod
    def _build_prompt(
        target_column: str,
        target_column_type: str,
        refine_with_existing_values_only: bool,
        value_counts: str,
        nl_prompt: str,
    ) -> str:
        if refine_with_existing_values_only:
            existing_values_prompt = "Do not introduce additional new unique values."
        else:
            existing_values_prompt = "You are allowed to introduce new unique values if necessary."

        prompt = f"""
        The data scientist wants to refine the data in the column '{target_column}' of type '{target_column_type}' 
        in a dataframe. The column has the following unique values with following counts: 
        {value_counts}. {existing_values_prompt}
        
        You need to assist the data scientists with writing a Python method `refine_column(df: pandas.DataFrame) -> pandas.DataFrame` to refine data in the column '{target_column}' from the pandas DataFrame `df`. 
        `refine_column(df: pandas.DataFrame) -> pandas.DataFrame` should return a modified dataframe without duplicates.
        You need to take care of refinement or near-refinement including finding semantic similarity, typos, or translation from one language to another.
        Additionally, you need to generalize column and remove outliers for better consistency.
        Mark outliers as 'unknown' or try to fix them if possible.
        The generated refinement/refinement code can build a correspondence dictionary on how to refine, union, and group values for better consistency and generalizability, use similarity, n-grams, tf-idf, or call external libraries for the refinement.
        Do not use pandas.drop_duplicates() and do not change the size of the input dataframe. If you construct corespondance dictionary, avoid duplicates.
        
        The data scientist wants you to take special care of the following: 
        {nl_prompt}.

        """

        prompt += f"""
        Return the codeblock that starts with "```python" and ends with ```end, and contains code to refine the column values using uniques values and existing methods. Optionally build a correspondence dictionary.
        
        IMPORTANT: Use the actual column name '{target_column}' in your code, not a placeholder like 'column_name'.
        
        Example 1: In a column 'categories', value counts are {{"sports shoes": 10, "sneakers": 5, "kicks": 1, "elefant": 10, "running shoes": 3}}. All shoes are for sports and can be refined into "sports shoes".
        Example 1 output:
        ```python
        def refine_column(df):
            column_name = 'categories'
            _refinement_correspondences = {{
                'sports shoes': 'sports shoes', 
                'sneakers': 'sports shoes', 
                'kicks': 'sports shoes', 
                'running shoes': 'sports shoes',
                'elefant': 'unknown',  # outlier
            }}
            df[column_name] = df[column_name].replace(_refinement_correspondences)
            return df
        ```end

        Example 2: In a column 'product_name', value counts are {{"Airpods": 3, "Kabellose Ohrhörer": 1, "Wireless Earbuds":10, "Écouteurs Sans": 1, "Auriculares Básicos": 1}}. All values are wireless earbuds and can be refined by translating into "Wireless Earbuds".
        Example 2 output:
        ```python
        def refine_column(df):
            column_name = 'product_name'
            _refinement_correspondences = {{
                'Airpods': 'Wireless Earbuds', 
                'Kabellose Ohrhörer': 'Wireless Earbuds', 
                'Wireless Earbuds': 'Wireless Earbuds', 
                'Écouteurs Sans': 'Wireless Earbuds',
                'Auriculares Básicos': 'Wireless Earbuds'
            }}
            df[column_name] = df[column_name].replace(_refinement_correspondences)
            return df
        ```end

        Example 3: In a column 'country_code', value counts are {{'DE': 2, 'US': 1, 'USA': 1, 'CAN': 1, 'cA ': 1, 'CA': 1, ' Cn': 1, 'CHN': 1, 'CN': 1, 'usa': 1, 'GER': 1, 'deu': 1, 'FRN': 1, 'FRA ': 1, 'fra': 1, 'FR': 1, 'GB': 1, 'UK ': 1, 'uk': 1, 'U-S': 1, ' UAS': 1, 'USA ': 1, 'UK': 1, 'DHL': 1}}. The values can be refined by removing types and multi-lingual instances from the data.
        Example 3 output:
        ```python
        def refine_column(df):
            column_name = 'country_code'
            normalization_map = {{
                "UAS": "US", "U-S": "US", "USA": "US",
                "GB": "UK",  # unify GB/UK
                "DEU": "DE", "GER": "DE", "GE": "DE",
                "FRA": "FR", "FRN": "FR",
                "CHN": "CN",
                "CAN": "CA",
                "DHL": "unknown",  # outlier
                "Dresden": "DE"  # outlier
            }}
            
            df[column_name] = df[column_name].astype(str).str.upper().str.strip()
            df[column_name] = df[column_name].replace(normalization_map)

            return df
        ```end

        Code formatting for your answer:
        ```python
        def refine_column(df):
            column_name = '{target_column}'
            # Code for the refinement
            ...
            return df
        ```end

        ENSURE YOU PRODUCE CORRECT RUNNABLE CODE. Codeblock:    
        """

        return prompt

    def fit(self, df: pd.DataFrame, y=None) -> Self:  # pylint: disable=unused-argument
        print(f"--- sempipes.sem_refine('{self.target_column}', '{self.nl_prompt}')")
        prompt_preview = self.nl_prompt[:40].replace("\n", " ").strip()

        if self._prefitted_state is not None:
            print(f"--- Using provided state for sempipes.sem_refine('{prompt_preview}...', '{self.target_column}')")
            self.generated_code_ = self._prefitted_state["generated_code"]

            # If state was provided but generated_code is empty, we still need to generate code
            if not self.generated_code_:
                print("\t> Warning: Provided state has empty generated_code, generating new code...")
            else:
                return self

        target_metric = "accuracy"
        if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
            target_metric = self._pipeline_summary.target_metric

        target_column_type = str(df[self.target_column].dtype)
        self.value_counts_ = json.dumps(df[self.target_column].value_counts().to_dict())

        prompt = self._build_prompt(
            self.target_column,
            target_column_type,
            self.refine_with_existing_values_only,
            self.value_counts_,
            self.nl_prompt,
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

                _try_to_execute(df, self.target_column, code_to_execute)

                generated_code.append(code)
                self.generated_code_ = generated_code
                break
            except Exception as e:  # pylint: disable=broad-except
                print(f"\t> An error occurred in attempt {attempt}:", e)
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to generate valid refinement code after {_MAX_RETRIES} attempts. "
                        f"Last error: {type(e).__name__}: {e}"
                    ) from e
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        f"Code: ```python{code}```\n Generate next feature (fixing error?):\n```python\n",
                    },
                ]

        if not self.generated_code_:
            raise RuntimeError(
                f"Failed to generate refinement code after {_MAX_RETRIES} attempts. " "No valid code was generated."
            )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ("generated_code_",))

        code_to_execute = "\n".join(self.generated_code_)
        refinement_func = safe_exec(code_to_execute, "refine_column")
        df = refinement_func(df)

        return df


def sem_refine(
    self: DataOp,
    target_column: str,
    nl_prompt: str,
    refine_with_existing_values_only: bool,
    name: str | None = None,
) -> DataOp:
    if name is None:
        name = f"random_{randint(0, 10000)}_name"

    data_op = self
    refinement_estimator = SemRefineWithLLM().generate_refinement_estimator(
        data_op=data_op,
        target_column=target_column,
        nl_prompt=nl_prompt,
        refine_with_existing_values_only=refine_with_existing_values_only,
        name=name,
    )
    return self.skb.apply(refinement_estimator, how="no_wrap").skb.set_name(name)
