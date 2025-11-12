import json
from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.operators.operators import EstimatorTransformer, SemRefineOperator


class SemRefineWithLLM(SemRefineOperator):
    def generate_refinement_estimator(
        self, target_column: str, nl_prompt: str, refine_with_existing_values_only: bool
    ) -> EstimatorTransformer:
        return LLMDeduplicator(
            target_column=target_column,
            nl_prompt=nl_prompt,
            refine_with_existing_values_only=refine_with_existing_values_only,
        )


_MAX_RETRIES = 5
_SYSTEM_PROMPT = (
    "You are a helpful assistant, assisting data scientists with improving their data preparation code and, for instance, refinement code. "
    "You answer only by generating code. Answer as concisely as possible."
)


def _try_to_execute(df: pd.DataFrame, target_column: str, code_to_execute: str) -> None:
    df_sample = df.head(50).copy(deep=True)

    refinement_func = safe_exec(code_to_execute, "refine_column")
    refined_sample_column = refinement_func(df_sample)

    assert (
        refined_sample_column.isna().sum() == df_sample[target_column].isna().sum()
    ).all(), "The refined column contains more NaN values."
    assert refined_sample_column.shape[0] == df_sample[target_column].shape[0], "The refined column shape was modified."
    print("\t> Code executed successfully on a sample dataframe.")


class LLMDeduplicator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_column: str,
        nl_prompt: str,
        refine_with_existing_values_only: bool,
    ) -> None:
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.refine_with_existing_values_only = refine_with_existing_values_only
        self.generated_code_: list[str] = []

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
        Do not use pandas.drop_duplicates() and do not change the size of the input dataframe.
        
        The data scientist wants you to take special care of the following: 
        {nl_prompt}.

        """

        prompt += """
        Return the codeblock that starts with "```python" and ends with ```end, and contains code to refine the column values using uniques values and existing methods. Optionally build a correspondence dictionary.
        
        Example 1: In a column 'categories', value counts are {"sports shoes": 10, "sneakers": 5, "kicks": 1, "elefant": 10, "running shoes": 3}. All shoes are for sports and can be refined into "sports shoes".
        Example 1 output:
        ```python
        def refine_column(df):
            _refinement_correspondences = {
                'sports shoes': 'sports shoes', 
                'sneakers': 'sports shoes', 
                'kicks': 'sports shoes', 
                'running shoes': 'sports shoes',
                'elefant': 'unknown', # outlier
            }
            df.loc[:, column_name] = df[:, column_name].replace(_refinement_correspondences)
            return df
        ```end

        Example 2: In a column 'product_name', value counts are {"Airpods": 3, "Kabellose Ohrhörer": 1, "Wireless Earbuds":10, "Écouteurs Sans": 1, "Auriculares Básicos": 1}. All values are wireless earbuds and can be refined by translating into "Wireless Earbuds".
        Example 2 output:
        ```python
        def refine_column(df):
            _refinement_correspondences = {
                'Airpods': 'Wireless Earbuds', 
                'Kabellose Ohrhörer': 'Wireless Earbuds', 
                'Wireless Earbuds': 'Wireless Earbuds', 
                'Écouteurs Sans': 'Wireless Earbuds',
                'Auriculares Básicos': 'Wireless Earbuds'
            }
            df.loc[:, column_name] = df[:, column_name].replace(_refinement_correspondences)
            return df
        ```end

        Example 3: In a column 'country_code', value counts are {'DE': 2, 'US': 1, 'USA': 1, 'CAN': 1, 'cA ': 1, 'CA': 1, ' Cn': 1, 'CHN': 1, 'CN': 1, 'usa': 1, 'GER': 1, 'deu': 1, 'FRN': 1, 'FRA ': 1, 'fra': 1, 'FR': 1, 'GB': 1, 'UK ': 1, 'uk': 1, 'U-S': 1, ' UAS': 1, 'USA ': 1, 'UK': 1, 'DHL': 1}. The values can be refined by removing types and multi-lingual instances from the data.
        Example 3 output:
        ```python
        def refine_column(df):
            normalization_map = {
                "UAS": "US", "U-S": "US", "USA": "US",
                "GB": "UK",  # unify GB/UK
                "DEU": "DE", "GER": "DE", "GE": "DE",
                "FRA": "FR", "FRN": "FR",
                "CHN": "CN",
                "CAN": "CA",
                "DHL": "unknown",  # outlier
                "Dresden": "DE"  # outlier
            }
            
            df[column_name] = df[column_name].str.upper().str.strip()
            df[:, column_name] = df[:, column_name].replace(normalization_map)

            return df
        ```end

        Code formatting for your answer:
        ```python
        # Code for the refinement
        ...
        ```end

        Codeblock:    
        """

        return prompt

    def fit(self, df: pd.DataFrame, y=None) -> Self:  # pylint: disable=unused-argument
        print(f"--- sempipes.sem_refine('{self.target_column}', '{self.nl_prompt}')")

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
                        self.refine_with_existing_values_only,
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
        refinement_func = safe_exec(code_to_execute, "refine_column")
        df = refinement_func(df)

        return df


def sem_refine(self: DataOp, target_column: str, nl_prompt: str, refine_with_existing_values_only: bool) -> DataOp:
    refinement_estimator = SemRefineWithLLM().generate_refinement_estimator(
        target_column, nl_prompt, refine_with_existing_values_only
    )
    return self.skb.apply(refinement_estimator)
