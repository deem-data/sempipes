import json
from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.operators.operators import EstimatorTransformer, SemDeduplicateOperator


class SemDeduplicateWithLLM(SemDeduplicateOperator):
    def generate_deduplication_estimator(
        self, target_column: str, nl_prompt: str, deduplicate_with_existing_values_only: bool
    ) -> EstimatorTransformer:
        return LLMDeduplicator(
            target_column=target_column,
            nl_prompt=nl_prompt,
            deduplicate_with_existing_values_only=deduplicate_with_existing_values_only,
        )


_MAX_RETRIES = 5
_SYSTEM_PROMPT = (
    "You are a helpful assistant, assisting data scientists with improving their data preparation code and, for instance, deduplication code. "
    "You answer only by generating code. Answer as concisely as possible."
)


def _try_to_execute(df: pd.DataFrame, target_column: str, code_to_execute: str) -> None:
    df_sample = df.head(50).copy(deep=True)

    deduplication_func = safe_exec(code_to_execute, "deduplicate_column")
    deduplicated_sample_column = deduplication_func(df_sample)

    assert (
        deduplicated_sample_column.isna().sum() == df_sample[target_column].isna().sum()
    ).all(), "The deduplicated column contains more NaN values."
    assert (
        deduplicated_sample_column.shape[0] == df_sample[target_column].shape[0]
    ), "The deduplicated column shape was modified."
    print("\t> Code executed successfully on a sample dataframe.")


class LLMDeduplicator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_column: str,
        nl_prompt: str,
        deduplicate_with_existing_values_only: bool,
    ) -> None:
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.deduplicate_with_existing_values_only = deduplicate_with_existing_values_only
        self.generated_code_: list[str] = []

    @staticmethod
    def _build_prompt(
        target_column: str,
        target_column_type: str,
        deduplicate_with_existing_values_only: bool,
        value_counts: str,
        nl_prompt: str,
    ) -> str:
        if deduplicate_with_existing_values_only:
            existing_values_prompt = "Do not introduce additional new unique values."
        else:
            existing_values_prompt = "You are allowed to introduce new unique values if necessary."

        prompt = f"""
        The data scientist wants to deduplicate the data in the column '{target_column}' of type '{target_column_type}' 
        in a dataframe. The column has the following unique values with following counts: 
        {value_counts}. {existing_values_prompt}
        
        You need to assist the data scientists with writing a Python method `deduplicate_column(df: pandas.DataFrame) -> pandas.DataFrame` to deduplicate data in the column '{target_column}' from the pandas DataFrame `df`. 
        `deduplicate_column(df: pandas.DataFrame) -> pandas.DataFrame` should return a modified dataframe without duplicates.
        Deduplication or near-deduplication can include finding semantic similarity, typos, or translation from one language to another.
        The generated deduplication code can build a correspondence dictionary on how to deduplicate, union, and group values for better consistency and generalizability, use similarity, n-grams, tf-idf, or call external libraries for the deduplication.
        Do not use pandas.drop_duplicates() and do not change the size of the input dataframe.
        
        The data scientist wants you to take special care to the following: 
        {nl_prompt}.

        """

        prompt += """
        Return the codeblock that starts with "```python" and ends with ```end, and contains code to deduplicate the column values using uniques values and existing methods. Optionally build a correspondence dictionary.
        
        Example 1: In a column 'categories', value counts are {"sports shoes": 10, "sneakers": 5, "kicks": 1, "running shoes": 3}. All shoes are for sports and can be deduplicated into "sports shoes".
        Example 1 output:
        ```python
        def deduplicate_column(df):
            _deduplication_correspondences = {
                'sports shoes': 'sports shoes', 
                'sneakers': 'sports shoes', 
                'kicks': 'sports shoes', 
                'running shoes': 'sports shoes'
            }
            df.loc[:, column_name] = df[:, column_name].replace(_deduplication_correspondences)
            return df
        ```end

        Example 2: In a column 'product_name', value counts are {"Airpods": 3, "Kabellose Ohrhörer": 1, "Wireless Earbuds":10, "Écouteurs Sans": 1, "Auriculares Básicos": 1}. All values are wireless earbuds and can be deduplicated by translating into "Wireless Earbuds".
        Example 2 output:
        ```python
        def deduplicate_column(df):
            _deduplication_correspondences = {
                'Airpods': 'Wireless Earbuds', 
                'Kabellose Ohrhörer': 'Wireless Earbuds', 
                'Wireless Earbuds': 'Wireless Earbuds', 
                'Écouteurs Sans': 'Wireless Earbuds',
                'Auriculares Básicos': 'Wireless Earbuds'
            }
            df.loc[:, column_name] = df[:, column_name].replace(_deduplication_correspondences)
            return df
        ```end

        Example 3: In a column 'country_code', value counts are {'DE': 2, 'US': 1, 'USA': 1, 'CAN': 1, 'cA ': 1, 'CA': 1, ' Cn': 1, 'CHN': 1, 'CN': 1, 'usa': 1, 'GER': 1, 'deu': 1, 'FRN': 1, 'FRA ': 1, 'fra': 1, 'FR': 1, 'GB': 1, 'UK ': 1, 'uk': 1, 'U-S': 1, ' UAS': 1, 'USA ': 1, 'UK': 1}. The values can be deduplicated by removing types and multi-lingual instances from the data.
        Example 3 output:
        ```python
        def deduplicate_column(df):
            normalization_map = {
                "UAS": "US", "U-S": "US", "USA": "US",
                "GB": "UK",  # unify GB/UK
                "DEU": "DE", "GER": "DE", "GE": "DE",
                "FRA": "FR", "FRN": "FR",
                "CHN": "CN",
                "CAN": "CA",
            }
            
            df[column_name] = df[column_name].str.upper().str.strip()
            df[:, column_name] = df[:, column_name].replace(normalization_map)

            return df
        ```end

        Code formatting for your answer:
        ```python
        # Code for the deduplication
        ...
        ```end

        Codeblock:    
        """

        return prompt

    def fit(self, df: pd.DataFrame, y=None) -> Self:  # pylint: disable=unused-argument
        print(f"--- sempipes.sem_deduplicate('{self.target_column}', '{self.nl_prompt}')")

        target_column_type = str(df[self.target_column].dtype)
        self.value_counts_ = json.dumps(df[self.target_column].value_counts().to_dict())

        messages = []
        for attempt in range(1, _MAX_RETRIES + 1):
            code = ""

            try:
                if attempt == 1:
                    prompt = self._build_prompt(
                        self.target_column,
                        target_column_type,
                        self.deduplicate_with_existing_values_only,
                        self.value_counts_,
                        self.nl_prompt,
                    )

                    messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

                code = generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(self.generated_code_)
                code_to_execute += "\n\n" + code

                _try_to_execute(df, self.target_column, code_to_execute)

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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ("generated_code_", "value_counts_"))

        code_to_execute = "\n".join(self.generated_code_)
        deduplication_func = safe_exec(code_to_execute, "deduplicate_column")
        df = deduplication_func(df)

        return df


def sem_deduplicate(
    self: DataOp, target_column: str, nl_prompt: str, deduplicate_with_existing_values_only: bool
) -> DataOp:
    deduplication_estimator = SemDeduplicateWithLLM().generate_deduplication_estimator(
        target_column, nl_prompt, deduplicate_with_existing_values_only
    )
    return self.skb.apply(deduplication_estimator)
