from collections.abc import Iterable
from typing import Self
from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.llm.llm import generate_python_code, generate_python_code_from_messages
from sempipes.operators.operators import SemFillNAOperator


class SemFillNALLLMPlusModel(SemFillNAOperator):
    def generate_imputation_estimator(self, data_op: DataOp, target_column: str, nl_prompt: str):
        # TODO explore computational graph or cached preview results to improve imputer generation
        return LearnedImputer(target_column, nl_prompt)


class LearnedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, nl_prompt: str):
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.imputation_model_: RandomForestClassifier | RandomForestRegressor | None = None

    def fit(self, df: pd.DataFrame, y=None) -> Self:
        print(f"--- sempipes.sem_fillna('{self.target_column}', '{self.nl_prompt}')")

        candidate_columns = [column for column in df.columns if column != self.target_column]
        
        y = df[self.target_column]
        known_mask = y.notna()
        df_train = df[known_mask].copy()
        
        print(f"\t> Fitting imputation model on columns {candidate_columns} of {known_mask.sum()} rows...")
        model = TabularPredictor(label=self.target_column).fit(df_train)
        
        self.imputation_model_ = model

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "imputation_model_")
        y = df[self.target_column]
        missing_mask = y.isna()
        num_missing_values = missing_mask.sum()

        if num_missing_values > 0:
            print(f"\t> Imputing {num_missing_values} values...")
            # predicted_values = self.imputation_model_.predict(df[missing_mask])  # type: ignore[union-attr]
            predicted_values = self.imputation_model_.predict(df[missing_mask].drop(columns=[self.target_column]))

            df.loc[missing_mask, self.target_column] = predicted_values
        return df
