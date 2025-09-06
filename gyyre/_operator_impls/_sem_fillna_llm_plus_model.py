from collections.abc import Iterable
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from gyyre._code_gen._exec import _safe_exec
from gyyre._code_gen._llm import _generate_python_code
from gyyre._operators import SemFillNAOperator


class SemFillNALLLMPlusModel(SemFillNAOperator):
    def generate_imputation_estimator(self, data_op: DataOp, target_column: str, nl_prompt: str):
        # TODO explore computational graph or cached preview results to improve imputer generation
        return LearnedImputer(target_column, nl_prompt)


class LearnedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, nl_prompt: str):
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.imputation_model_ = None

    @staticmethod
    def _build_prompt(
        target_column: str, target_column_type: str, candidate_columns: Iterable[str], nl_prompt: str
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
        candidate_columns = [column for column in df.columns if column != self.target_column]

        prompt = self._build_prompt(self.target_column, target_column_type, candidate_columns, self.nl_prompt)
        python_code = _generate_python_code(prompt)
        feature_columns = _safe_exec(python_code, "__chosen_columns")

        X = df[feature_columns]
        y = df[self.target_column]

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
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
        print(f"\t> Fitting imputation model {learner} on columns {feature_columns} of {known_mask.sum()} rows...")
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
            df.loc[missing_mask, self.target_column] = self.imputation_model_.predict(df[missing_mask])
        return df
