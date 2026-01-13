import json

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sempipes.llm.llm import (
    batch_generate_json_retries,
    get_generic_message,
)
from sempipes.logging import get_logger
from sempipes.operators.sem_extract_features._shared import (
    SYSTEM_PROMPT,
    Modality,
    _get_modality,
    build_output_columns_to_generate_llm,
    create_file_url,
)

logger = get_logger()


_MAX_RETRIES = 5


def _get_pre_post_feature_generation_messages(columns_to_generate: list[str]) -> tuple[str, str]:
    task_prompt = f"""
    Your goal is to help a data scientist generate features from multi-modal data.
    You are provided the name of the features to extract, how to extract them, and from which columns.

    Generate the following columns: {columns_to_generate}.

    """

    example_result = {key: "..." for key in columns_to_generate}
    example_result_str = json.dumps(example_result, indent=2)

    response_example = f"""
    Please respond with a JSON object with feature names as keys and generated feature values as values. 
    JSON object should contain the following keys: {columns_to_generate}. 

    Answer with only the JSON, without any additional text, explanations, comments, or formatting.
    
    Example response:
    ```json
    {example_result_str}
    ```
    """

    return task_prompt, response_example


class LLMFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, nl_prompt: str, input_columns: list[str], output_columns: dict[str, str] | None) -> None:
        self.nl_prompt = nl_prompt
        self.input_columns = input_columns
        self.output_columns = {} if output_columns is None else output_columns
        self.modality_per_column: dict[str, Modality] = {}
        self.generated_features_: list[dict[str, object]] = []

    def _build_mm_generation_prompt(self, row):
        audios, images = [], []
        context_prompt = """
        Here are values of columns (images/text/audio) that can be used for the feature extraction:
        """
        # Build value prompt and process input columns
        value_prompt_parts = []
        for i, feature in enumerate(self.generated_features_):
            feature_name = feature["feature_name"]
            feature_prompt = feature["feature_prompt"]
            input_columns = feature["input_columns"]
            value_prompt_parts.append(
                f"\n {i+1}. Generate value of a column named `{feature_name}`. {feature_prompt}. "
                f"For generation you use the follwing context and columns: {input_columns}."
            )
            # Process each input column
            self._process_input_columns(row, input_columns, context_prompt, audios, images)

        value_prompt = "".join(value_prompt_parts)
        pre_message, post_message = _get_pre_post_feature_generation_messages(
            columns_to_generate=list(self.output_columns.keys())
        )

        context = [
            {"type": "text", "text": pre_message + value_prompt + post_message + context_prompt},
        ]
        context += audios
        context += images

        return context

    def _process_input_columns(self, row, input_columns, context_prompt, audios, images):
        """Process input columns and populate context, audios, and images lists."""
        for input_column in input_columns:
            modality = self.modality_per_column[input_column]
            if modality == Modality.TEXT:
                context_prompt += f"\nTextual column `{input_column}`: {row[input_column]}."
            elif modality == Modality.AUDIO:
                audio_data_url, audio_format = create_file_url(row[input_column])
                # litellm expects raw base64 string, not data URL
                # Extract base64 from data URL: "data:audio/wav;base64,<base64>" -> "<base64>"
                audio_base64 = (
                    audio_data_url.split(",", 1)[1]
                    if audio_data_url.startswith("data:") and "," in audio_data_url
                    else audio_data_url
                )
                audios.append({"type": "input_audio", "input_audio": {"data": audio_base64, "format": audio_format}})
            elif modality == Modality.IMAGE:
                image, _ = create_file_url(row[input_column])
                images.append({"type": "image_url", "image_url": {"url": image}})

    def fit(self, df: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        # Determine modalities of input columns
        self.modality_per_column = {column: _get_modality(df[column]) for column in self.input_columns}

        if self.output_columns == {}:
            # Ask LLM which features to generate given input columns
            logger.info(f"sempipes.sem_extract_features('{self.input_columns}', '{self.nl_prompt}')")
            self.generated_features_, self.output_columns = build_output_columns_to_generate_llm(
                df, self.input_columns, self.nl_prompt
            )
        else:
            # Use user-given columns
            logger.info(
                f"sempipes.sem_extract_features('{self.input_columns}', '{self.output_columns}', '{self.nl_prompt}')"
            )

            for new_feature, prompt in self.output_columns.items():
                self.generated_features_.append(
                    {"feature_name": new_feature, "feature_prompt": prompt, "input_columns": self.input_columns}
                )

        logger.info(f"Generated possible columns: {self.generated_features_}")

        return self

    def extract_features_with_llm(self, df):
        # Construct prompts with multi-modal data
        prompts = []
        for _, row in df.iterrows():
            prompt = self._build_mm_generation_prompt(row=row)
            prompts.append(get_generic_message(SYSTEM_PROMPT, prompt))

        encoded_results = batch_generate_json_retries(prompts)
        generated_columns = {}

        for idx, encoded_result in enumerate(encoded_results):
            if encoded_result is None:
                logger.warning(f"LLM returned None for row {idx}, using default value")
                # Use default value for failed requests
                for feature in self.generated_features_:
                    column_name = feature["feature_name"]
                    default_value = "Unknown"  # Default fallback value
                    generated_columns[column_name] = (
                        [default_value]
                        if generated_columns.get(column_name) is None
                        else generated_columns[column_name] + [default_value]
                    )
                continue

            try:
                parsed_result = json.loads(encoded_result)
                assert isinstance(parsed_result, dict), f"Expected a dict encoded in JSON, but got: {encoded_result}"

                for column_name, cell_value in parsed_result.items():
                    generated_columns[column_name] = (
                        [cell_value]
                        if generated_columns.get(column_name) is None
                        else generated_columns[column_name] + [cell_value]
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                logger.error(f"Error processing response for row {idx}: {encoded_result}", exc_info=True)
                # Use default value for parsing failures
                for feature in self.generated_features_:
                    column_name = feature["feature_name"]
                    default_value = "Unknown"  # Default fallback value
                    generated_columns[column_name] = (
                        [default_value]
                        if generated_columns.get(column_name) is None
                        else generated_columns[column_name] + [default_value]
                    )

        # Assign new results back
        for new_column, new_vals in generated_columns.items():
            df[new_column] = new_vals

        logger.info(f"Generated {len(generated_columns)} columns: {list(generated_columns.keys())}.")

        return df

    def transform(self, df):
        check_is_fitted(self, "generated_features_")
        return self.extract_features_with_llm(df=df)
