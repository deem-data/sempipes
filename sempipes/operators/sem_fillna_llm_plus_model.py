import inspect
from collections.abc import Iterable
from typing import Any, Self

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.operators.operators import SemFillNAOperator

_MAX_RETRIES = 5
_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates code for the data imputation model. Answer only with the Python code."
)


class SemFillNALLLMPlusModel(SemFillNAOperator):
    def generate_imputation_estimator(self, data_op: DataOp, target_column: str, nl_prompt: str):
        return LearnedImputer(target_column, nl_prompt)


def _expects_dataframe(model) -> bool:
    """Check if model expects DataFrame input."""
    if isinstance(model, Pipeline):
        return any(
            isinstance(step[1], ColumnTransformer) or (isinstance(step[1], Pipeline) and _expects_dataframe(step[1]))
            for step in model.steps
        )

    class_name = model.__class__.__name__
    if "AutoGluon" in class_name or "Imputer" in class_name:
        return True

    fit_method = getattr(model, "fit", None)
    if not fit_method:
        return False

    # Check type hints
    try:
        sig = inspect.signature(fit_method)
        for param in sig.parameters.values():
            if param.annotation and "DataFrame" in str(param.annotation):
                return True
    except (ValueError, TypeError):
        pass

    # Check source code
    try:
        source = inspect.getsource(fit_method)
        if "DataFrame" in source and ("isinstance" in source or "TypeError" in source):
            return True
    except (OSError, TypeError):
        pass

    return False


def _create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical columns."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
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


def _wrap_model(model: Any, preprocess: ColumnTransformer) -> Any:
    """Wrap model with preprocessing if needed."""
    if _expects_dataframe(model):
        return model

    if isinstance(model, Pipeline):
        has_preprocessing = any(isinstance(step[1], ColumnTransformer) for step in model.steps)
        return model if has_preprocessing else Pipeline([("prep", preprocess), ("pipeline", model)])

    return Pipeline([("prep", preprocess), ("est", model)])


class LearnedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, nl_prompt: str):
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.imputation_model_: Any = None
        self.feature_columns_: list[str] = []
        self._training_columns_: list[str] | None = None

    @staticmethod
    def _build_prompt(
        target_column: str, target_column_type: str, candidate_columns: Iterable[str], nl_prompt: str
    ) -> str:
        return f"""
The data scientist wants to fill missing values in the column '{target_column}' of type '{target_column_type}' 
in a dataframe. The dataframe has the following columns available: {candidate_columns}. 

Choose which columns to use to fill the missing values. Special care: {nl_prompt}.

Code formatting:
```python
__chosen_columns = [<subset of `candidate_columns`>]
```end
"""

    @staticmethod
    def _build_model_prompt(
        target_column: str,
        target_column_type: str,
        selected_columns: Iterable[str],
        nl_prompt: str,
        is_numeric_target: bool,
    ) -> str:
        target_type = "numeric" if is_numeric_target else "categorical"
        try:
            from autogluon.tabular import __version__ as ag_version  # pylint: disable=import-outside-toplevel
        except ImportError:
            ag_version = "unknown"

        return f"""
Suggest five very different models for missing value imputation in column '{target_column}' (type: {target_column_type}, {target_type}).
Available columns: {selected_columns}. Special care: {nl_prompt}.

Use sklearn models (RandomForestRegressor, LinearRegression, etc.) or imputers (IterativeImputer) or AutoGluon (TabularPredictor). If you use AutoGluon's TabularPredictor, you need to set `time_limit` in `fit` to 900 seconds.
Models must have `fit_transform` or `fit`/`predict` methods.

Check that you use correct inputs for the models, e.g., pandas DataFrames vs numpy arrays.

IMPORTANT: The data has ALREADY been preprocessed (scaled/encoded) before being passed to these models. You MUST NOT add any further preprocessing, feature selection, or column-based transformations.
Versions: sklearn {sklearn.__version__}, pandas {pd.__version__}, autogluon {ag_version}.
You are NOT allowed to use `tempfile` library.

```python
model1 = ...  # Missing data imputation model
model2 = ...  # Missing data imputation model
model3 = ...  # Missing data imputation model
model4 = ...  # Missing data imputation model
model5 = ...  # Missing data imputation model
suggested_models_and_imputers = [model1, model2, model3, model4, model5]
```end

Codeblock:
"""

    def fit(self, df: pd.DataFrame, y=None) -> Self:  # pylint: disable=too-many-locals,too-many-statements
        print(f"--- sempipes.sem_fillna('{self.target_column}', '{self.nl_prompt}')")

        # Select feature columns
        target_column_type = str(df[self.target_column].dtype)
        candidate_columns = [c for c in df.columns if c != self.target_column]
        # prompt = self._build_prompt(self.target_column, target_column_type, candidate_columns, self.nl_prompt)
        # feature_columns = safe_exec(generate_python_code(prompt), "__chosen_columns")
        self.feature_columns_ = candidate_columns

        X = df[self.feature_columns_]
        y = df[self.target_column]
        preprocess = _create_preprocessor(X)
        is_numeric_target = pd.api.types.is_numeric_dtype(y)

        # Generate models
        prompt = self._build_model_prompt(
            self.target_column, target_column_type, self.feature_columns_, self.nl_prompt, is_numeric_target
        )
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        generated_code: list[str] = []

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                code = generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(generated_code) + "\n\n" + code
                print(code_to_execute)

                models = safe_exec(code_to_execute, "suggested_models_and_imputers")
                suggested_models_and_imputers = [_wrap_model(m, preprocess) for m in models]
                generated_code.append(code)
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Attempt {attempt} failed: {e}")
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed: {type(e).__name__}: {e}\nCode: ```python{code}```\nPlease generate corrected code:\n```python\n",
                    },
                ]

        # Evaluate models
        known_mask = y.notna()
        known_indices = np.where(known_mask)[0]
        train_idx, val_idx = train_test_split(np.arange(len(known_indices)), test_size=0.2, random_state=42)
        train_indices = known_indices[train_idx]
        val_indices = known_indices[val_idx]

        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        # For DataFrame-based models
        y_all_nan = y.iloc[np.concatenate([train_indices, val_indices])].copy()
        y_all_nan.iloc[len(train_indices) :] = np.nan
        Xy_all_nan = pd.concat(
            [X.iloc[np.concatenate([train_indices, val_indices])], y_all_nan.to_frame(name=self.target_column)], axis=1
        )

        scores = []
        for idx, model_candidate in enumerate(suggested_models_and_imputers, 1):
            try:
                is_pipeline = isinstance(model_candidate, Pipeline)
                actual_model = model_candidate.steps[-1][1] if is_pipeline else model_candidate
                is_dataframe_based = _expects_dataframe(actual_model)

                if hasattr(model_candidate, "fit") and hasattr(model_candidate, "predict") and not is_dataframe_based:
                    model_candidate.fit(X_train, y_train)
                    y_pred = model_candidate.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    scores.append(rmse)
                elif hasattr(model_candidate, "fit_transform") or is_dataframe_based:
                    Xy_transformed = model_candidate.fit_transform(Xy_all_nan)
                    y_pred_all = (
                        Xy_transformed[self.target_column].values
                        if isinstance(Xy_transformed, pd.DataFrame)
                        else (Xy_transformed[:, -1] if Xy_transformed.ndim > 1 else Xy_transformed)
                    )
                    y_pred_val = y_pred_all[len(train_indices) :]
                    scores.append(np.sqrt(mean_squared_error(y_val.values, y_pred_val)))
                else:
                    scores.append(float("inf"))
                print(f"RMSE for model {idx}: {scores[-1]}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error evaluating model {idx}: {e}")
                scores.append(float("inf"))

        # Select and fit best model
        best_idx = np.argmin(scores)
        model = suggested_models_and_imputers[best_idx]
        print(f"Selected model {best_idx + 1} with RMSE: {scores[best_idx]}")
        print(f"\t> Fitting on {known_mask.sum()} rows...")

        is_pipeline = isinstance(model, Pipeline)
        actual_model = model.steps[-1][1] if is_pipeline else model
        is_dataframe_based = _expects_dataframe(actual_model)

        if is_dataframe_based:
            # DataFrame-based models need DataFrame with features + target
            Xy_train = pd.concat([X[known_mask], y[known_mask].to_frame(name=self.target_column)], axis=1)
            model.fit(Xy_train)
            # Store the columns used for training (needed for transform)
            self._training_columns_ = list(Xy_train.columns)
        else:
            # Standard sklearn models expect separate X and y
            model.fit(X[known_mask], y[known_mask])
            self._training_columns_ = None

        self.imputation_model_ = model
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "imputation_model_")
        y = df[self.target_column]
        missing_mask = y.isna()

        if missing_mask.sum() == 0:
            return df

        print(f"\t> Imputing {missing_mask.sum()} values...")
        df_missing = df[missing_mask].copy()
        is_dataframe_based = _expects_dataframe(self.imputation_model_)

        if hasattr(self.imputation_model_, "predict"):
            # For DataFrame-based models, use only the columns they were trained on
            if is_dataframe_based and hasattr(self, "_training_columns_") and self._training_columns_:
                X_input = df_missing[self._training_columns_]
            else:
                X_input = df_missing
            result = self.imputation_model_.predict(X_input)  # type: ignore[union-attr]
            # Handle both Series and array outputs
            if isinstance(result, pd.Series):
                predicted_values = result.values
            else:
                predicted_values = result
        elif hasattr(self.imputation_model_, "transform"):
            # For transform-based models (like IterativeImputer), use only training columns
            if is_dataframe_based and hasattr(self, "_training_columns_") and self._training_columns_:
                X_input = df_missing[self._training_columns_]
            else:
                X_input = df_missing
            transformed = self.imputation_model_.transform(X_input)  # type: ignore[union-attr]
            if isinstance(transformed, pd.DataFrame):
                predicted_values = transformed[self.target_column].values
            else:
                predicted_values = transformed[:, -1] if transformed.ndim > 1 else transformed
        else:
            raise ValueError("Model must have either 'predict' or 'transform' method")

        df.loc[missing_mask, self.target_column] = predicted_values
        return df
