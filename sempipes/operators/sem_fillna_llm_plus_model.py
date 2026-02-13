import inspect
import re
from collections.abc import Iterable
from typing import Any, Self

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
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
    # Only AutoGluon-style predictors accept mixed-type DataFrames. sklearn Imputers expect numeric array input.
    if "AutoGluon" in class_name:
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


def _create_preprocessor(X: pd.DataFrame, use_ordinal: bool = False) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical columns."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    cat_encoder = (
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        if use_ordinal
        else ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        cat_encoder,
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def _get_target_output_slice_and_categories(
    preprocess: ColumnTransformer, target_column: str
) -> tuple[slice, list[Any] | None]:
    """Return (slice into preprocessor output for target column, categories or None if numeric)."""
    start = 0
    for _name, trans, cols in preprocess.transformers_:
        if not hasattr(trans, "transform"):
            continue
        for i, col in enumerate(cols):
            width = 1
            categories = None
            if _name == "cat" and hasattr(trans, "named_steps"):
                oh = trans.named_steps.get("oh")
                ord_enc = trans.named_steps.get("ord")
                if oh is not None and hasattr(oh, "categories_") and i < len(oh.categories_):
                    categories = list(oh.categories_[i])
                    width = len(categories)
                elif ord_enc is not None and hasattr(ord_enc, "categories_") and i < len(ord_enc.categories_):
                    categories = list(ord_enc.categories_[i])
                    width = 1
            if col == target_column:
                return slice(start, start + width), categories
            start += width
    return slice(-1, None), None  # fallback: last column


def _wrap_model(model: Any, preprocess: ColumnTransformer, preprocess_for_imputer: ColumnTransformer) -> Any:
    """Wrap model with preprocessing if needed."""
    if _expects_dataframe(model):
        return model

    use_preprocess = preprocess_for_imputer if hasattr(model, "fit_transform") else preprocess

    if isinstance(model, Pipeline):
        has_preprocessing = any(isinstance(step[1], ColumnTransformer) for step in model.steps)
        return model if has_preprocessing else Pipeline([("prep", use_preprocess), ("pipeline", model)])

    return Pipeline([("prep", use_preprocess), ("est", model)])


class LearnedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, nl_prompt: str):
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.imputation_model_: Any = None
        self.feature_columns_: list[str] = []
        self._training_columns_: list[str] | None = None
        self._target_output_slice_: slice | None = None
        self._target_categories_: list[Any] | None = None

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

Use sklearn models (RandomForestRegressor, LinearRegression, etc.) or imputers (IterativeImputer) or AutoGluon (TabularPredictor). 

CRITICAL 1: If you use AutoGluon's TabularPredictor, you need to set `time_limit` in `fit` to 10 seonds if user asks for a very fast method, otherwise set it to 300 seconds.

CRITICAL 2: Models must have `fit_transform` or `fit`/`predict` methods.

CRITICAL 3: Do NOT write any training code. Only instantiate models. The training and evaluation will be handled elsewhere.

Check that you use correct inputs for the models, e.g., pandas DataFrames vs numpy arrays.

Ensure that you can handle both numeric and categorical columns that contain text.

IMPORTANT 1: The data has ALREADY been preprocessed (scaled/encoded) before being passed to these models. You MUST NOT add any further preprocessing, feature selection, or column-based transformations.
IMPORTANT1b: Do NOT call `fit`, `fit_transform`, `predict`, or `transform`. Only instantiate model objects. Do NOT create dummy dataframes or use `df` in the code.
Versions: sklearn {sklearn.__version__}, pandas {pd.__version__}, autogluon {ag_version}.
You are NOT allowed to use `tempfile` library.

IMPORTANT 2: Always import IterativeImputer via `from sklearn.experimental import enable_iterative_imputer` first.
```python
from sklearn.experimental import enable_iterative_imputer # Always import IterativeImputer via this first
from sklearn.impute import IterativeImputer

model1 = ...  # Missing data imputation model
model2 = ...  # Missing data imputation model
model3 = ...  # Missing data imputation model
model4 = ...  # Missing data imputation model
model5 = ...  # Missing data imputation model
suggested_models_and_imputers = [model1, model2, model3, model4, model5]
```end

Codeblock:
"""

    def fit(self, df: pd.DataFrame, y=None) -> Self:  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        print(f"--- sempipes.sem_fillna('{self.target_column}', '{self.nl_prompt}')")

        # Select feature columns
        target_column_type = str(df[self.target_column].dtype)
        candidate_columns = [c for c in df.columns if c != self.target_column]
        self.feature_columns_ = candidate_columns

        X = df[self.feature_columns_]
        y = df[self.target_column]
        # Include target so preprocessor encodes it (needed for IterativeImputer and categorical targets)
        Xy = pd.concat([X, y.to_frame(name=self.target_column)], axis=1)
        preprocess = _create_preprocessor(Xy, use_ordinal=False)
        preprocess_for_imputer = _create_preprocessor(Xy, use_ordinal=True)
        is_numeric_target = pd.api.types.is_numeric_dtype(y)

        # Generate models
        prompt = self._build_model_prompt(
            self.target_column, target_column_type, self.feature_columns_, self.nl_prompt, is_numeric_target
        )
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        generated_code: list[str] = []
        suggested_models_and_imputers = None
        last_exception = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                code = generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(generated_code) + "\n\n" + code
                print(code_to_execute)

                forbidden = [
                    r"\bfit_transform\(",
                    r"\bfit\(",
                    r"\bpredict\(",
                    r"\btransform\(",
                    r"\bdf\b",
                ]
                if any(re.search(pattern, code) for pattern in forbidden):
                    raise ValueError(
                        "Generated code must only instantiate model objects (no fit/predict/transform or df usage)."
                    )

                models = safe_exec(code_to_execute, "suggested_models_and_imputers")
                suggested_models_and_imputers = [_wrap_model(m, preprocess, preprocess_for_imputer) for m in models]
                generated_code.append(code)
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_exception = e
                print(f"Attempt {attempt} failed: {e}")
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed: {type(e).__name__}: {e}\nCode: ```python{code}```\nPlease generate corrected code:\n```python\n",
                    },
                ]

        if suggested_models_and_imputers is None:
            raise RuntimeError(
                f"sem_fillna failed after {_MAX_RETRIES} attempts. "
                "Generated code must define 'suggested_models_and_imputers' and run without errors. "
                "Ensure models are trained only on rows where the target column is not missing "
                "(e.g. dropna(subset=[target]) before fit)."
            ) from last_exception

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
        for model_index, model_candidate in enumerate(suggested_models_and_imputers, 1):
            try:
                is_pipeline = isinstance(model_candidate, Pipeline)
                actual_model = model_candidate.steps[-1][1] if is_pipeline else model_candidate
                is_dataframe_based = _expects_dataframe(actual_model)

                if hasattr(model_candidate, "fit") and hasattr(model_candidate, "predict") and not is_dataframe_based:
                    model_candidate.fit(X_train, y_train)
                    y_pred = model_candidate.predict(X_val)
                    if is_numeric_target:
                        score = np.sqrt(mean_squared_error(y_val, y_pred))
                    else:
                        score = 1.0 - accuracy_score(y_val, y_pred)
                    scores.append(score)
                elif hasattr(model_candidate, "fit_transform") or is_dataframe_based:
                    Xy_transformed = model_candidate.fit_transform(Xy_all_nan)
                    if isinstance(Xy_transformed, pd.DataFrame):
                        y_pred_all = Xy_transformed[self.target_column].values
                    elif is_pipeline and hasattr(model_candidate.steps[0][1], "transformers_"):
                        sl, cats = _get_target_output_slice_and_categories(
                            model_candidate.steps[0][1], self.target_column
                        )
                        target_out = Xy_transformed[:, sl] if Xy_transformed.ndim > 1 else Xy_transformed
                        if cats is not None and target_out.ndim > 1:
                            idx = np.argmax(target_out, axis=1)
                            y_pred_all = np.array([cats[i] for i in idx])
                        else:
                            y_pred_all = target_out.ravel() if target_out.ndim > 1 else target_out
                    else:
                        y_pred_all = Xy_transformed[:, -1] if Xy_transformed.ndim > 1 else Xy_transformed
                    y_pred_val = y_pred_all[len(train_indices) :]
                    if is_numeric_target:
                        score = np.sqrt(mean_squared_error(y_val.values, y_pred_val))
                    else:
                        score = 1.0 - accuracy_score(y_val.values, y_pred_val)
                    scores.append(score)
                else:
                    scores.append(float("inf"))
                print(f"Score for model {model_index}: {scores[-1]}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error evaluating model {model_index}: {e}")
                scores.append(float("inf"))

        # Select and fit best model
        best_idx = np.argmin(scores)
        model = suggested_models_and_imputers[best_idx]
        print(f"Selected model {best_idx + 1} with score: {scores[best_idx]}")
        print(f"\t> Fitting on {known_mask.sum()} rows...")

        is_pipeline = isinstance(model, Pipeline)
        actual_model = model.steps[-1][1] if is_pipeline else model
        is_dataframe_based = _expects_dataframe(actual_model)
        Xy_train = pd.concat([X[known_mask], y[known_mask].to_frame(name=self.target_column)], axis=1)

        if is_dataframe_based:
            # AutoGluon-style: DataFrame with features + target
            model.fit(Xy_train)
            self._training_columns_ = list(Xy_train.columns)
            self._target_output_slice_ = None
            self._target_categories_ = None
        elif is_pipeline and hasattr(actual_model, "fit_transform"):
            # Pipeline (preprocess + imputer): pass Xy so preprocessor encodes target and imputer sees it
            model.fit(Xy_train)
            self._training_columns_ = list(Xy_train.columns)
            prep = model.steps[0][1]
            self._target_output_slice_, self._target_categories_ = _get_target_output_slice_and_categories(
                prep, self.target_column
            )
        else:
            # Standard fit(X, y) predictors
            model.fit(X[known_mask], y[known_mask])
            self._training_columns_ = None
            self._target_output_slice_ = None
            self._target_categories_ = None

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
        if hasattr(self.imputation_model_, "predict"):
            # For DataFrame-based models, use only the columns they were trained on
            if hasattr(self, "_training_columns_") and self._training_columns_:
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
            if hasattr(self, "_training_columns_") and self._training_columns_:
                X_input = df_missing[self._training_columns_]
            else:
                X_input = df_missing
            transformed = self.imputation_model_.transform(X_input)  # type: ignore[union-attr]
            if isinstance(transformed, pd.DataFrame):
                predicted_values = transformed[self.target_column].values
            elif hasattr(self, "_target_output_slice_") and self._target_output_slice_ is not None:
                sl = self._target_output_slice_
                target_out = transformed[:, sl] if transformed.ndim > 1 else transformed
                if self._target_categories_ is not None and target_out.ndim > 1:
                    idx = np.argmax(target_out, axis=1)
                    predicted_values = np.array([self._target_categories_[i] for i in idx])
                else:
                    predicted_values = target_out.ravel() if target_out.ndim > 1 else target_out
            else:
                predicted_values = transformed[:, -1] if transformed.ndim > 1 else transformed
        else:
            raise ValueError("Model must have either 'predict' or 'transform' method")

        df.loc[missing_mask, self.target_column] = predicted_values
        return df
