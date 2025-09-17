import base64
import json
import mimetypes
from enum import Enum

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from gyyre._llm._llm import (
    _batch_generate_results,
    _generate_json_from_messages,
    _get_generic_message,
    _unwrap_json,
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
    modality_per_column: dict[str, Modality],
) -> tuple[str, dict, dict]:
    samples_str = ""
    samples_image: dict[str, list[tuple[str, str]]] = {}
    samples_audio: dict[str, list[tuple[str, str]]] = {}

    for column, modality in modality_per_column.items():
        samples_list = df.loc[df[column].notna(), column].head(2).tolist()

        if modality == Modality.TEXT:
            samples_str += f"Samples of column `{column}`: {samples_list}.\n"

        elif modality == Modality.AUDIO:
            samples_audio[column] = [_create_file_url(val) for val in samples_list]

        elif modality == Modality.IMAGE:
            samples_audio[column] = [_create_file_url(val) for val in samples_list]

    return samples_str, samples_image, samples_audio


def _get_feature_suggestion_message(
    nl_prompt: str, input_columns: list[str], samples_text: str, samples_image: dict, samples_audio: dict
) -> list[dict]:
    # TODO add column description from scrub

    task_prompt = f"""
    Your goal is to help a data scientist select which features can be generated from multi-modal data from a pandas dataframe for a machine learning script which they are developing. 
    You can make your decision for each column individually or for combinations of multiple columns. 

    The data scientist wants you to take special care to the following: {nl_prompt}

    The data scientist wants to generate new features based on the following columns in a dataframe: {input_columns}.

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

    content: list[dict] = [
        {"type": "text", "text": task_prompt},
    ]

    if samples_text != "":
        content.append({"type": "text", "text": f"Samples of text columns: \n{samples_text}"})

    for column, images in samples_image.items():
        content.append({"type": "text", "text": f"Samples of image column `{column}`:"})
        for image in images:
            content.append({"type": "image", "image": image})

    for column, (audios, audio_formats) in samples_audio.items():
        content.append({"type": "text", "text": f"Samples of image column `{column}`:"})
        for audio, audio_format in zip(audios, audio_formats):
            content.append({"type": "input_audio", "input_audio": {"data": audio, "format": audio_format}})

    content.append({"type": "text", "text": response_example})

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": content}]

    return messages


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

    Answer with only the JSON, without any additional text, explanations, or formatting.
    
    Example response:
    ```json
    {example_result_str}
    ```
    """

    return task_prompt, response_example


def _get_modality(column: pd.Series) -> Modality:
    sample_of_column = column.sample(frac=0.2)
    modality = None
    if (
        pd.api.types.is_string_dtype(sample_of_column)
        and sample_of_column.str.endswith(r"\.(jpg|jpeg|png|bmp|gif)$", na=False).all()
    ):
        modality = Modality.IMAGE
    elif (
        pd.api.types.is_string_dtype(sample_of_column)
        and sample_of_column.str.endswith(r"\.(wav|mp3|flac|aac|ogg)$", na=False).all()
    ):
        modality = Modality.AUDIO
    else:
        modality = Modality.TEXT
    return modality


class LLMFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, nl_prompt: str, input_columns: list[str], output_columns: dict[str, str] | None) -> None:
        self.nl_prompt = nl_prompt
        self.input_columns = input_columns
        self.output_columns = {} if output_columns is None else output_columns
        self.modality_per_column: dict[str, Modality] = {}
        self.generated_features_: list[dict[str, object]] = []

    def _build_mm_generation_prompt(self, row):
        audios, images = [], []
        value_prompt = ""
        context_prompt = """
        Here are values of columns (images/text/audio) that can be used for the feature extraction:
        """
        for i, feature in enumerate(self.generated_features_):
            feature_name = feature["feature_name"]
            feature_prompt = feature["feature_prompt"]
            input_columns = feature["input_columns"]

            value_prompt = f"""\n {i+1}. Generate value of a column named `{feature_name}`. {feature_prompt}. 
            For generation you use the follwing context and columns: {input_columns}.
            """

            for input_column in input_columns:
                modality = self.modality_per_column[input_column]
                if modality == Modality.TEXT:
                    context_prompt += f"\nTextual column `{input_column}`: {row[input_column]}."

                elif modality == Modality.AUDIO:
                    audio, audio_format = _create_file_url(row[input_column])
                    audios.append({"type": "input_audio", "input_audio": {"data": audio, "format": audio_format}})

                elif modality == Modality.IMAGE:
                    image, _ = _create_file_url(row[input_column])
                    images.append({"type": "image_url", "image_url": {"url": image}})

        pre_message, post_message = _get_pre_post_feature_generation_messages(
            columns_to_generate=list(self.output_columns.keys())
        )

        context = [
            {"type": "text", "text": pre_message + value_prompt + post_message + context_prompt},
        ]
        context += audios
        context += images

        return context

    def _build_output_columns_to_generate_llm(self, df: pd.DataFrame):
        # Form prompts with examples
        # TODO can be replaced with column descritions later
        samples_str, samples_image, samples_audio = _get_modality_prompts(
            df=df, modality_per_column=self.modality_per_column
        )
        messages = _get_feature_suggestion_message(
            nl_prompt=self.nl_prompt,
            input_columns=self.input_columns,
            samples_text=samples_str,
            samples_image=samples_image,
            samples_audio=samples_audio,
        )

        generated_output = _generate_json_from_messages(messages=messages)

        # Validate that generated dicts have correct keys for later generation
        for feature in json.loads(generated_output):
            if list(feature.keys()) == ["feature_name", "feature_prompt", "input_columns"]:
                self.generated_features_.append(feature)
                self.output_columns[feature["feature_name"]] = feature["feature_prompt"]
            else:
                print("\t> Unable to parse some of the suggested features.")

    def fit(self, df: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        # Determine modalities of input columns
        self.modality_per_column = {column: _get_modality(df[column]) for column in self.input_columns}

        if self.output_columns == {}:
            # Ask LLm which features to generate given input columns
            print(f"--- gyyre.sem_extract_features('{self.input_columns}', '{self.nl_prompt}')")

            self._build_output_columns_to_generate_llm(df)

        else:
            # Use user-given columns
            print(
                f"--- gyyre.sem_extract_features('{self.input_columns}', '{self.output_columns}', '{self.nl_prompt}')"
            )

            for new_feature, prompt in self.output_columns.items():
                self.generated_features_.append(
                    {"feature_name": new_feature, "feature_prompt": prompt, "input_columns": self.input_columns}
                )

        print(f"\t> Generated possible columns: { self.generated_features_}")
        return self

    def transform(self, df):
        check_is_fitted(self, "generated_features_")

        # Construct prompts with multi-modal data
        prompts = []
        for _, row in df.iterrows():
            prompt = self._build_mm_generation_prompt(row=row)
            prompts.append(_get_generic_message(_SYSTEM_PROMPT, prompt))

        results = _batch_generate_results(prompts, batch_size=100)

        # Parse new results
        generated_columns: dict = {}
        for result in results:
            raw_result = _unwrap_json(result)

            try:
                for column_name, cell_value in json.loads(raw_result).items():
                    generated_columns[column_name] = (
                        [cell_value]
                        if generated_columns.get(column_name) is None
                        else generated_columns[column_name] + [cell_value]
                    )
            except Exception as e:
                print(f"Error processing response: {raw_result}")
                raise e

        # Assign new results back
        for new_column, new_vals in generated_columns.items():
            df[new_column] = new_vals

        print(f"\t> Generated {len(generated_columns)} columns: {list(generated_columns.keys())}")

        return df


class SemExractFeaturesLLM(SemExtractFeaturesOperator):
    def generate_features_extractor(
        self, nl_prompt: str, input_columns: list[str], output_columns: dict[str, str] | None
    ) -> EstimatorTransformer:
        return LLMFeatureExtractor(nl_prompt=nl_prompt, input_columns=input_columns, output_columns=output_columns)
