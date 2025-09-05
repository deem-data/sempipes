from typing import Self
from collections.abc import Iterable
from gyyre._operators import SemFillNAOperator
from gyyre._code_gen._llm import _batch_generate_results_from_prompts

from gyyre._code_gen._exec import _safe_exec

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


class SemFillNAWithLLLM(SemFillNAOperator):
    def generate_imputation_estimator(
        self,
        data_op,
        target_column,
        nl_prompt,
        with_existing_vals=True,
    ):
        return LLMImputer(
            target_column=target_column,
            nl_prompt=nl_prompt,
            with_existing_vals=with_existing_vals,
        )


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
