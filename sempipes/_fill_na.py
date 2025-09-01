from sempipes._inspection import context_graph
from sempipes.code_gen._llm import _generate_python_code
from sempipes.code_gen._exec import _safe_exec

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def _impute_with_model(df, target_column, feature_columns):
    X = df[feature_columns]
    y = df[target_column]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocess predictors ONLY (not the target):
    # - impute predictors' own missingness (median for numeric, most_frequent for cats)
    # - scale numeric; one-hot encode categoricals
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    is_numeric_target = pd.api.types.is_numeric_dtype(y)
    if is_numeric_target:
        learner = RandomForestRegressor(random_state=0)
    else:
        learner = RandomForestClassifier(random_state=0)

    model = Pipeline([("prep", preprocess),("est", learner)])

    # Train on rows where target is known
    known_mask = y.notna()
    print(f"\tTraining imputation model {learner} on columns {feature_columns} of {known_mask.sum()} rows...")
    model.fit(X[known_mask], y[known_mask])

    # Predict missing target values
    missing_mask = y.isna()
    print(f"\tImputing {missing_mask.sum()} values...")
    df.loc[missing_mask, target_column] = model.predict(X[missing_mask])

def _build_prompt(target_column, target_column_type, candidate_columns, nl_prompt):
    return f"""
    The data scientist wants to fill missing values in the column '{target_column}' of type '{target_column_type}' in a dataframe. The dataframe has the following columns available to help with this task: {candidate_columns}. You need to assist the data scientists with choosing which columns to use to fill the missing values in the target column.
    
    The data scientist wants you to take special care to the following: {nl_prompt}.
    
    Code formatting for your answer:
    ```python
    __chosen_columns = [<subset of `candidate_columns`>]
    ```end

    The codeblock ends with ```end and starts with "```python"
Codeblock:    
"""

def _internal_sem_fillna(data_op, target_column, nl_prompt):
    ctx = context_graph(data_op._skrub_impl)
    def sempipes_sem_fillna(df):
        print(f"--- Sempipes.sem_fillna('{target_column}', '{nl_prompt}')")

        target_column_type = str(df[target_column].dtype)
        candidate_columns = [column for column in df.columns if column != target_column]

        prompt = _build_prompt(target_column, target_column_type, candidate_columns, nl_prompt)
        python_code = _generate_python_code(prompt)
        feature_columns = _safe_exec(python_code, "__chosen_columns")

        _impute_with_model(df, target_column, feature_columns)

        return df
    return sempipes_sem_fillna

def sem_fillna(self, target_column, nl_prompt):
    return self.skb.apply_func(_internal_sem_fillna(self, target_column, nl_prompt))