from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.logging import get_logger
from sempipes.operators.operators import ContextAwareMixin, OptimisableMixin
from sempipes.operators.sem_extract_features._shared import SYSTEM_PROMPT, build_output_columns_to_generate_llm

logger = get_logger()


_MAX_RETRIES = 5


def _add_memorized_history(
    memory: list[dict[str, Any]] | None,
    messages: list[dict[str, str]],
    target_metric: str,
) -> None:
    if memory is not None and len(memory) > 0:
        current_score = None

        messages += [
            {
                "role": "user",
                "content": """
                IMPORTANT: Try to generate code that extracts better features for the downstream model. Here are some things that you can try:

                    - Try to use a pretrained model specialized for the domain of the data.
                    - Try different variants of pretrained models for the feature extraction.
                    - Adjust the hyperparameters of previously chosen pretrained models for the feature extraction.
                    - Try models with a larger maximum sequence length.
                    - Try models with a larger number of parameters.

                Below is a history of the code that has been generated and executed so far, together with the performance of the model.

                IMPORTANT: Explain your reasoning for the code changes you made in comments of the code.
                """,
            }
        ]

        for memory_line in memory:
            memorized_code = memory_line["update"]
            memorized_score = memory_line["score"]

            if current_score is None:
                improvement = abs(memorized_score)
            else:
                improvement = memorized_score - current_score

            if improvement > 0.0:
                positive_impact_sentence = (
                    "The code was executed and improved the downstream performance. "
                    "You may choose to copy from this previous version of the code for the next version of the code."
                )
                current_score = memorized_score
            else:
                positive_impact_sentence = (
                    f"The last code changes did not improve performance. " f"(Improvement: {improvement})"
                )

            messages += [
                {"role": "assistant", "content": memorized_code},
                {
                    "role": "user",
                    "content": f"Performance for last code block: {target_metric}={memorized_score:.5f}. "
                    f".{positive_impact_sentence}\nNext codeblock:\n",
                },
            ]


def _get_code_feature_generation_message(
    columns_to_generate: list[str], column_descriptions, features_to_extract: list[dict]
) -> str:
    column_description_prompt = ""
    for column, properties in column_descriptions.items():
        column_description_prompt += (
            f"Column {column}\n"
            f" - Dtype: {properties['dtype']}\n"
            f" - Modality: {properties['modality']}\n"
            f" - Samples: {properties['samples']}\n"
        )
        if properties["modality"] == "text":
            column_description_prompt += f" - Max words per value: {properties['max_words']}\n"
            column_description_prompt += f" - Mean words per value: {properties['mean_words']}\n"
        column_description_prompt += "\n"

    task_prompt = f"""
Your goal is to help a data scientist generate Python code for the feature generation/extraction from multi-modal data. You need to extract information from text, image, or audio data. 
You can use any models from `transformers` library that can be used zero-shot, without additional fine-tuning. You are allowed to leverage modality-specific models for each modality.

You are provided the name of the features to extract, how to extract them, and from which columns within a pandas DataFrame `df`.

Generate Python code with method `extract_features(df: pd.DataFrame) -> pd.DataFrame` for feature extraction using `transformers`, `torch` libraries. 
`extract_features` takes the original DataFrame `df`.
`extract_features` returns the original dataframe with newly generated columns: {columns_to_generate}.

Use the following columns as input for the feature extraction:

{column_description_prompt}

In the code, try to use the least loaded GPU if multiple GPUs are available. Prefer MPS (Metal Performance Shaders) for Apple Silicon devices if available, then CUDA, then CPU.

"""
    code_example = f"""
Code formatting for each added column:
```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    pipeline,
)


def pick_device():
    # Prefer MPS for Apple Silicon, then CUDA, then CPU
    if torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        # Find GPU with most free memory
        free = []
        for i in range(torch.cuda.device_count()):
            f, t = torch.cuda.mem_get_info(i)
            free.append((f, i))
        free.sort(reverse=True)
        _, idx = free[0]
        print(f"Chosen GPU: {{idx}}")
        return torch.device(f"cuda:{{idx}}")
    else:
        print("Using CPU")
        return torch.device("cpu")

device = pick_device()

features_to_extract = {features_to_extract}

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # Extract features using transformers and other libraries
    # IMPORTANT: When creating a pipeline, handle the device parameter correctly:
    # - For CUDA: use device=device.index (e.g., device=0 for cuda:0)
    # - For MPS: use device=device (the device object itself, not device.index as MPS doesn't have index)
    # - For CPU: use device=-1 or omit device parameter entirely (some versions prefer device=None)
    # Example: pipe = pipeline(..., device=device if device.type != "cpu" else -1)
    # Or: pipe = pipeline(..., device=0 if device.type == "cuda" else (device if device.type == "mps" else -1))
    ...

    # Add features to the original df as new columns
    return df
```end

"""
    post_message = """

IMPORTANT:
 - Add comments to the code which explain the rationale for choosing a particular pretrained model.
 - Make sure that the code feeds the data in batches to the GPU for efficiency.
 - Use the `tqdm` library to show a progress bar for the feature extraction.

Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""
    return task_prompt + code_example + post_message


def _try_to_execute(df: pd.DataFrame, code_to_execute: str, generated_columns: list[str]) -> None:
    df_sample = df.head(50).copy(deep=True)

    feature_extraction_func = safe_exec(code_to_execute, "extract_features")
    extracted_sample = feature_extraction_func(df_sample)

    expected_columns = generated_columns + list(df.columns)

    assert all(
        expected_column in extracted_sample.columns for expected_column in expected_columns
    ), f"The returned DataFrame does not contain all required columns. Expected: {expected_columns}. Actual: {list(extracted_sample.columns)} "

    logger.debug("Code executed successfully on a sample dataframe.")


def _describe_column(column: pd.Series):
    column_description = {
        "dtype": column.dtype,
        "modality": "text",
        "samples": column.sample(n=10, random_state=42).tolist(),
    }

    sample_of_column = column.sample(frac=0.2, random_state=42)
    if (
        pd.api.types.is_string_dtype(sample_of_column)
        and sample_of_column.str.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"), na=False).all()
    ):
        column_description["modality"] = "image"
    elif (
        pd.api.types.is_string_dtype(sample_of_column)
        and sample_of_column.str.endswith(("wav", "mp3", "flac", "aac", "ogg"), na=False).all()
    ):
        column_description["modality"] = "audio"

    if column_description["modality"] == "text":
        # Remove NaN values for processing
        clean_col = column.dropna()
        # Compute number of words per value
        word_counts = clean_col.astype(str).apply(lambda x: len(x.split(" ")))
        column_description["max_words"] = word_counts.max() if not word_counts.empty else 0
        column_description["mean_words"] = word_counts.mean() if not word_counts.empty else 0

    return column_description


class CodeBasedFeatureExtractor(BaseEstimator, TransformerMixin, ContextAwareMixin, OptimisableMixin):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        nl_prompt: str,
        input_columns: list[str],
        output_columns: dict[str, str] | None,
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
    ) -> None:
        self.nl_prompt = nl_prompt
        self.input_columns = input_columns
        self.output_columns_not_given = output_columns is None
        self.output_columns = {} if output_columns is None else output_columns
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory
        self._inspirations: list[dict[str, Any]] | DataOp | None = _inspirations
        self.generated_code_: str | None = None

    def empty_state(self):
        state = {
            "generated_code": """
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    return df        
"""
        }

        if self.output_columns_not_given:
            state["generated_features"] = [
                {"feature_name": "", "feature_prompt": self.nl_prompt, "input_columns": self.input_columns}
            ]

        return state

    def state_after_fit(self):
        state = {"generated_code": self.generated_code_}
        if self.output_columns_not_given:
            state["generated_features"] = self.output_columns

        return state

    def memory_update_from_latest_fit(self):
        if self.generated_code_ is not None:
            return self.generated_code_
        return OptimisableMixin.EMPTY_MEMORY_UPDATE

    def fit(self, df: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        if self._prefitted_state is not None:
            self.generated_code_ = self._prefitted_state["generated_code"]
            if "generated_features" in self._prefitted_state:
                self.output_columns = self._prefitted_state["generated_features"]
            return self

        if self.output_columns == {}:
            _, self.output_columns = build_output_columns_to_generate_llm(df, self.input_columns, self.nl_prompt)

        logger.info(f"sempipes.sem_extract_features('{self.input_columns}', '{list(self.output_columns.keys())}')")

        column_descriptions = {column: _describe_column(df[column]) for column in self.input_columns}

        features_to_extract = []
        for new_feature, prompt in self.output_columns.items():
            features_to_extract.append(
                {"feature_name": new_feature, "feature_prompt": prompt, "input_columns": self.input_columns}
            )

        self.synthesize_extraction_code(df, column_descriptions, features_to_extract)

        return self

    def synthesize_extraction_code(self, df, column_descriptions, features_to_extract):
        # Construct prompts with multi-modal data
        prompt = _get_code_feature_generation_message(
            columns_to_generate=list(self.output_columns.keys()),
            column_descriptions=column_descriptions,
            features_to_extract=features_to_extract,
        )

        target_metric = "accuracy"
        if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
            target_metric = self._pipeline_summary.target_metric

        # Generate code for multi-modal data
        messages = []
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if attempt == 1:
                    messages += [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                    _add_memorized_history(self._memory, messages, target_metric)

                code = generate_python_code_from_messages(messages)

                # Try to extract actual features
                _try_to_execute(
                    df,
                    code,
                    generated_columns=list(self.output_columns.keys()),
                )

                self.generated_code_ = code
                break
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f"An error occurred in attempt {attempt}: {e}", exc_info=True)
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        f"Code: ```python{code}```\n Generate next feature (fixing error?):\n```python\n",
                    },
                ]

    def transform(self, df):
        check_is_fitted(self, "generated_code_")

        code_to_execute = self.generated_code_
        feature_extraction_func = safe_exec(code_to_execute, "extract_features")

        logger.info(f"Extracting features from {len(df)} rows")
        df_with_features = feature_extraction_func(df.copy(deep=True))
        return df_with_features
