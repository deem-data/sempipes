<<<<<<< HEAD
from typing import Self
from collections.abc import Iterable
=======
import time
from gyyre._operators import SemFillNAOperator
from gyyre._code_gen._llm import (
    _batch_generate_results_from_prompts,
    _generate_result_from_prompt,
)
from gyyre._code_gen._exec import _safe_exec

>>>>>>> 292e6ed (Add LLM and sem_fillna)
import pandas as pd
import numpy as np

from skrub import DataOp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class SemFillNALLLMPlusModel(SemFillNAOperator):
    def generate_imputation_estimator(
        self,
        data_op,
        target_column,
        nl_prompt,
        with_existing_vals=True,
        fill_with_llm=False,
    ):
        # TODO explore computational graph or cached preview results to improve imputer generation
        if fill_with_llm:
            return LLMImputer(
                target_column=target_column,
                nl_prompt=nl_prompt,
                with_existing_vals=with_existing_vals,
            )
        else:
            return LearnedImputer(target_column, nl_prompt)


class LearnedImputer(BaseEstimator, TransformerMixin):

    def __init__(self, target_column: str, nl_prompt: str):
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.imputation_model_ = None

    @staticmethod
    def _build_prompt(
        target_column: str,
        target_column_type: str,
        candidate_columns: Iterable[str],
        nl_prompt: str
    ) -> str:
        return f"""
        The data scientist wants to fill missing values in the column '{target_column}' of type '{target_column_type}' 
        in a dataframe. The dataframe has the following columns available to help with this task: 
        {candidate_columns}. 
        
        You need to assist the data scientists with choosing which columns to use to fill the missing values in 
        the target column. The data scientist wants you to take special care to the following: 
        {nl_prompt}.

        Code formatting for your answer:
        ```python
        __chosen_columns = [<subset of `candidate_columns`>]
        ```end

        The codeblock ends with ```end and starts with "```python"
    Codeblock:    
    """

    def fit(self, df: pd.DataFrame, y=None) -> Self:

        print(f"--- gyyre.sem_fillna('{self.target_column}', '{self.nl_prompt}')")

        target_column_type = str(df[self.target_column].dtype)
        candidate_columns = [
            column for column in df.columns if column != self.target_column
        ]

        prompt = self._build_prompt(
            self.target_column, target_column_type, candidate_columns, self.nl_prompt
        )
        python_code = _generate_result_from_prompt(prompt, generate_code=True)
        feature_columns = _safe_exec(python_code, "__chosen_columns")

        X = df[feature_columns]
        y = df[self.target_column]

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("oh", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )

        is_numeric_target = pd.api.types.is_numeric_dtype(y)
        if is_numeric_target:
            learner = RandomForestRegressor(random_state=0)
        else:
            learner = RandomForestClassifier(random_state=0)

        model = Pipeline([("prep", preprocess), ("est", learner)])

        # TODO we could keep a small holdout set to measure the imputer performance
        # Train on rows where target is known
        known_mask = y.notna()
        print(
            f"\t> Fitting imputation model {learner} on columns {feature_columns} of {known_mask.sum()} rows..."
        )
        model.fit(X[known_mask], y[known_mask])
        self.imputation_model_ = model

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "imputation_model_")
        y = df[self.target_column]
        missing_mask = y.isna()
        num_missing_values = missing_mask.sum()

        if num_missing_values > 0:
            print(f"\t> Imputing {num_missing_values} values...")
            df.loc[missing_mask, self.target_column] = self.imputation_model_.predict(
                df[missing_mask]
            )
        return df


class LLMImputer(BaseEstimator, TransformerMixin):

    def __init__(self, target_column, nl_prompt, with_existing_vals=True):
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.with_existing_vals = with_existing_vals
        self.imputation_model_ = None

    @staticmethod
    def _build_prompt(
        target_column,
        target_column_type,
        candidate_columns,
        nl_prompt,
        target_column_unique_values,
        with_existing_vals=True,
    ):
        return (
            f"""
        The data scientist wants to fill missing values in the column '{target_column}' of type '{target_column_type}' in a dataframe. 
        The dataframe has the following columns available with these values to help with this task: {candidate_columns}. You need to assist the data scientist with filling the missing values in the target column."""
            + f""" The target column has the following unique values: {target_column_unique_values}. Use only existing values to fill in the missing values. """
            if with_existing_vals
            else f""" Example values from the {target_column} column: {target_column_unique_values}. """
            + f""" The data scientist wants you to take special care to the following: {nl_prompt}.
        Answer with only the value to impute, without any additional text or formatting.
        Result:    
        """
        )

    def fit(self, df, y=None):
        # TODO maybe add examples of good imputations
        print(
            f"--- Sempipes.sem_fillna_llm('{self.target_column}', '{self.nl_prompt}')"
        )

        return self

    def transform(self, df):
        ix_to_impute = df[self.target_column].isna()
        rows_to_impute = df[ix_to_impute]

        if rows_to_impute.shape[0] == 0:
            print("\t> No missing values to impute.")
            return df

        target_column_type = str(df[self.target_column].dtype)

        # Get target unique values or examples
        target_unique_vals = df[self.target_column].unique()
        if not self.with_existing_vals and len(target_unique_vals) > 5:
            target_unique_vals = target_unique_vals[:5]
        target_column_unique_values = ", ".join(map(str, target_unique_vals))

        # Build prompts for each row to impute
        prompts, row_indices = [], []
        for i, row in rows_to_impute.iterrows():
            candidate_columns = {
                column: row[column]
                for column in df.columns
                if column != self.target_column
            }
            prompt = self._build_prompt(
                self.target_column,
                target_column_type,
                candidate_columns,
                self.nl_prompt,
                target_column_unique_values,
                self.with_existing_vals,
            )

            row_indices.append(i)
            prompts.append(prompt)

        results = _batch_generate_results_from_prompts(prompts, generate_code=False)

        df.loc[ix_to_impute, self.target_column] = results

        print(f"\t> Imputed {len(results)} values...")

        return df
