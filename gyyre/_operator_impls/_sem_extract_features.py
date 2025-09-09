import base64
import json
import mimetypes
from enum import Enum
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from gyyre._code_gen._llm import (
    _batch_generate_results,
    _generate_json_from_messages,
    _get_generic_message,
    _unwrap_llm_response,
)
from gyyre._operators import EstimatorTransformer, SemExtractFeaturesOperator

_SYSTEM_PROMPT = """
You are an expert data scientist, assisting feature extraction from the multi-modal data such as text/images/audio.
"""


# class syntax
class Modality(Enum):
    TEXT = 1
    AUDIO = 2
    IMAGE = 3


def _create_file_url(local_path: str) -> tuple[str, str]:
    """
    Converts a local image or audio file into a base64 data URL.

    Supports common image formats (png, jpg, jpeg, gif) and audio formats (wav, mp3, mpeg).
    """
    # Guess the MIME type from file extension
    mime_type, _ = mimetypes.guess_type(local_path)
    if mime_type is None:
        raise ValueError(f"Unsupported file type for {local_path}")

    with open(local_path, "rb") as f:
        file_bytes = f.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")

    # Construct data URL
    data_url = f"data:{mime_type};base64,{file_base64}"
    return data_url, mime_type


def _get_modality_prompts(
    df: pd.DataFrame,
    col_modality_dict: dict[str, Modality],
) -> tuple[str, dict, dict]:
    samples_str = ""
    samples_image: dict[str, list[tuple[str, str]]] = {}
    samples_audio: dict[str, list[tuple[str, str]]] = {}

    for col, modality in col_modality_dict.items():
        samples_list = df.loc[df[col].notna(), col].head(2).tolist()

        if modality == Modality.TEXT:
            samples_str += f"Samples of column `{col}`: {samples_list}.\n"

        elif modality == Modality.AUDIO:
            samples_audio[col] = [_create_file_url(val) for val in samples_list]

        elif modality == Modality.IMAGE:
            samples_audio[col] = [_create_file_url(val) for val in samples_list]

    return samples_str, samples_image, samples_audio


def _get_feature_suggestion_message(
    nl_prompt: str, input_cols: list[str], samples_text: str, samples_image: dict, samples_audio: dict
) -> list[object]:
    # TODO add column description from scrub

    task_prompt = f"""
    Your goal is to help a data scientist select which features can be generated from multi-modal data from a pandas dataframe for a machine learning script 
        which they are developing. You can your decision for each column individually or for combinations of multiple columns. 

    The data scientist wants you to take special care to the following: {nl_prompt}

    The data scientist wants to generate new features based on the following columns in a dataframe: {input_cols}.

    Here is detailed information about a column for which you need to make a decision now:
    """

    response_example = """
    Please respond with a JSON object per each feature to extract. JSON object should contain the following keys: `feature_name` - name of the new feature to generate, `feature_prompt` - prompt to generate this feature using an LLM, and `input_columns` - list of columns to use as input to generate this feature. 
    
    Example response for a column `image_column`:
    ```json
    [
        {
            "feature_name": "image_brightness",
            "feature_prompt": "Generate a categorical feature that represents color of the product on the image.",
            "input_columns": ["image_column"]
        }
    ]
    ```
    """

    content: list[dict[str, Any]] = [
        {"type": "text", "text": task_prompt},
    ]

    if samples_text != "":
        content.append({"type": "text", "text": f"Samples of text columns: \n{samples_text}"})

    for col, images in samples_image.items():
        content.append({"type": "text", "text": f"Samples of image column `{col}`:"})
        for img in images:
            content.append({"type": "image", "image": img})

    for col, (audios, audio_formats) in samples_audio.items():
        content.append({"type": "text", "text": f"Samples of image column `{col}`:"})
        for audio, audio_format in zip(audios, audio_formats):
            content.append({"type": "input_audio", "input_audio": {"data": audio, "format": audio_format}})

    content.append({"type": "text", "text": response_example})

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": content}]

    return messages


def _get_pre_post_feature_generation_messages(cols_to_generate: list[str]) -> tuple[str, str]:
    task_prompt = f"""
    Your goal is to help a data scientist generate features from multi-modal data.
    You are provided the name of the features to extract, how to extract them, and from which columns.

    Generate the following columns: {cols_to_generate}.

    """

    example_result = {key: "..." for key in cols_to_generate}
    example_result_str = json.dumps(example_result, indent=2)

    response_example = f"""
    Please respond with a JSON object with feature names as keys and generated feature values as values. 
    JSON object should contain the following keys: {cols_to_generate}. 

    Answer with only the JSON, without any additional text, explanations, or formatting.
    
    Example response:
    ```json
    {example_result_str}
    ```
    """

    return task_prompt, response_example


def _get_modality(col: pd.Series) -> Modality:
    sample_col = col.sample(frac=0.2)
    modality = None
    if (
        pd.api.types.is_string_dtype(sample_col)
        and sample_col.str.endswith(r"\.(jpg|jpeg|png|bmp|gif)$", na=False).all()
    ):
        modality = Modality.IMAGE
    elif (
        pd.api.types.is_string_dtype(sample_col)
        and sample_col.str.endswith(r"\.(wav|mp3|flac|aac|ogg)$", na=False).all()
    ):
        modality = Modality.AUDIO
    else:
        modality = Modality.TEXT
    return modality


class LLMFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, nl_prompt: str, input_cols: list[str], output_cols: dict[str, str] | None) -> None:
        self.nl_prompt = nl_prompt
        self.input_cols = input_cols
        self.output_cols = {} if output_cols is None else output_cols
        self.col_modality_dict: dict[str, Modality] = {}
        self.generated_features_list_: list[dict[str, object]] = []

    def _build_mm_generation_prompt(self, row):
        audios, images = [], []
        value_prompt = ""
        context_prompt = """
        Here are values of columns (images/text/audio) that can be used for the feature extraction:
        """
        for i, feature_dict in enumerate(self.generated_features_list_):
            feature_name = feature_dict["feature_name"]
            feature_prompt = feature_dict["feature_prompt"]
            input_columns = feature_dict["input_columns"]

            value_prompt = f"""\n {i+1}. Generate value of a column named `{feature_name}`. {feature_prompt}. 
            For generation you use the follwing context and columns: {input_columns}.
            """

            for col in input_columns:
                modality = self.col_modality_dict[col]
                if modality == Modality.TEXT:
                    context_prompt += f"\nTextual column `{col}`: {row[col]}."

                elif modality == Modality.AUDIO:
                    audio, audio_format = _create_file_url(row[col])
                    audios.append({"type": "input_audio", "input_audio": {"data": audio, "format": audio_format}})

                elif modality == Modality.IMAGE:
                    image, _ = _create_file_url(row[col])
                    images.append({"type": "image_url", "image_url": {"url": image}})

        pre_message, post_message = _get_pre_post_feature_generation_messages(
            cols_to_generate=list(self.output_cols.keys())
        )

        context = [
            {"type": "text", "text": pre_message + value_prompt + post_message + context_prompt},
        ]
        context += audios
        context += images

        return context

    def _build_output_cols_to_generate_llm(self, df: pd.DataFrame):
        # Form prompts with examples
        # TODO can be replaced with column descritions later
        samples_str, samples_image, samples_audio = _get_modality_prompts(
            df=df, col_modality_dict=self.col_modality_dict
        )
        messages = _get_feature_suggestion_message(
            nl_prompt=self.nl_prompt,
            input_cols=self.input_cols,
            samples_text=samples_str,
            samples_image=samples_image,
            samples_audio=samples_audio,
        )

        generated_output_json = _generate_json_from_messages(messages=messages)

        # Validate that generated dicts have correct keys for later generation
        for feature_dict in json.loads(generated_output_json):
            if list(feature_dict.keys()) == ["feature_name", "feature_prompt", "input_columns"]:
                self.generated_features_list_.append(feature_dict)
                self.output_cols[feature_dict["feature_name"]] = feature_dict["feature_prompt"]
            else:
                print("\t> Unable to parse some of the suggested features.")

    def fit(self, df: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        # Determine modalities of input columns
        self.col_modality_dict = {col: _get_modality(df[col]) for col in self.input_cols}

        if self.output_cols == {}:
            # Ask LLm which features to generate given input columns
            print(f"--- Sempipes.sem_extract_features('{self.input_cols}', '{self.nl_prompt}')")

            self._build_output_cols_to_generate_llm(df)

        else:
            # Use user-given columns
            print(f"--- Sempipes.sem_extract_features('{self.input_cols}', '{self.output_cols}', '{self.nl_prompt}')")

            for new_feature, prompt in self.output_cols.items():
                self.generated_features_list_.append(
                    {"feature_name": new_feature, "feature_prompt": prompt, "input_columns": self.input_cols}
                )

        print(f"\t> Generated possible columns: { self.generated_features_list_}")
        return self

    def transform(self, df):
        check_is_fitted(self, "generated_features_list_")

        # Construct prompts with multi-modal data
        prompts = []
        for _, row in df.iterrows():
            prompt = self._build_mm_generation_prompt(row=row)
            prompts.append(_get_generic_message(_SYSTEM_PROMPT, prompt))

        result_jsons = _batch_generate_results(prompts, batch_size=100)

        # Parse new results
        generated_cols: dict[str, list[Any]] = {}
        for res in result_jsons:
            raw_res = _unwrap_llm_response(res, prefix="```json", suffix="```", suffix2="```end")

            try:
                for k, v in json.loads(raw_res).items():
                    generated_cols[k] = [v] if generated_cols.get(k) is None else generated_cols[k] + [v]
            except Exception as e:
                print(f"Error processing response: {raw_res}")
                raise e

        # Assign new results back
        for new_col, new_vals in generated_cols.items():
            df[new_col] = new_vals

        print(f"\t> Generated {len(generated_cols)} columns: {list(generated_cols.keys())}")

        return df


class SemExractFeaturesLLM(SemExtractFeaturesOperator):
    def generate_features_extractor(
        self, nl_prompt: str, input_cols: list[str], output_cols: dict[str, str] | None
    ) -> EstimatorTransformer:
        return LLMFeatureExtractor(nl_prompt=nl_prompt, input_cols=input_cols, output_cols=output_cols)
