import base64
import json
import mimetypes
from enum import Enum

import pandas as pd

from sempipes.llm.llm import generate_json_from_messages
from sempipes.logging import get_logger

logger = get_logger()


SYSTEM_PROMPT = """
You are an expert data scientist, assisting feature extraction from the multi-modal data such as text/images/audio.
"""


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


def create_file_url(local_path: str) -> tuple[str, str]:
    """
    Converts a local image or audio file into a base64 data URL.

    Supports common image formats (png, jpg, jpeg, gif) and audio formats (wav, mp3, mpeg).
    """

    # Guess the MIME type from file extension
    mime_type, _ = mimetypes.guess_type(local_path)
    if mime_type is None:
        raise ValueError(f"Unsupported file type for {local_path}")

    data_type = mime_type.split("/")[1]
    if local_path.startswith("http://") or local_path.startswith("https://"):
        return local_path, data_type

    with open(local_path, "rb") as f:
        file_bytes = f.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")

    # Construct data URL
    data_url = f"data:{mime_type};base64,{file_base64}"

    return data_url, data_type


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
            samples_audio[column] = [create_file_url(audio_sample) for audio_sample in samples_list]

        elif modality == Modality.IMAGE:
            samples_image[column] = [create_file_url(image_sample) for image_sample in samples_list]

    return samples_str, samples_image, samples_audio


def _get_feature_suggestion_message(
    nl_prompt: str, input_columns: list[str], samples_text: str, samples_image: dict, samples_audio: dict
) -> list[dict]:
    task_prompt = f"""
    Your goal is to help a data scientist select which features can be generated from multi-modal data from a pandas dataframe for a machine learning script which they are developing. 
    You can make your decision for each column individually or for combinations of multiple columns. 

    The data scientist wants you to take special care of the following: {nl_prompt}

    The data scientist wants to generate new features based on the following columns in a dataframe: {input_columns}.

    Here is detailed information about a column for which you need to make a decision now:
    """

    response_example = """
    Please respond with a JSON object per each feature to extract. JSON object should contain the following keys: `feature_name` - name of the new feature to generate, `feature_prompt` - prompt to generate this feature using an LLM, and `input_columns` - list of columns to use as input to generate this feature. 
    
    Example response for columns ["description", "image"]:
    ```json
    [
        {
            "feature_name": "image_brightness",
            "feature_prompt": "Generate a categorical feature that represents color of the product on the image.",
            "input_columns": ["description"]
        }
    ]
    ```

    Answer ONLY with a JSON object like in the example.
    """

    content: list[dict] = [
        {"type": "text", "text": task_prompt},
    ]

    if samples_text != "":
        content.append({"type": "text", "text": f"Samples of text columns: \n{samples_text}"})

    for column, images in samples_image.items():
        content.append({"type": "text", "text": f"Samples of image column `{column}`:"})
        for image, _ in images:
            content.append({"type": "image_url", "image_url": {"url": image}})

    for column, audios in samples_audio.items():
        content.append({"type": "text", "text": f"Samples of audio column `{column}`:"})
        for audio_data_url, audio_format in audios:
            # litellm expects raw base64 string
            if isinstance(audio_data_url, str) and audio_data_url.startswith("data:"):
                # Extract base64 part after the comma
                audio_base64 = audio_data_url.split(",", 1)[1] if "," in audio_data_url else audio_data_url
            else:
                audio_base64 = audio_data_url
            content.append({"type": "input_audio", "input_audio": {"data": audio_base64, "format": audio_format}})

    content.append({"type": "text", "text": response_example})

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    return messages


def _get_modality(column: pd.Series) -> Modality:
    """Determine the modality of a column based on its content."""
    sample_of_column = column.sample(frac=0.2)
    if (
        pd.api.types.is_string_dtype(sample_of_column)
        and sample_of_column.str.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"), na=False).all()
    ):
        return Modality.IMAGE

    if (
        pd.api.types.is_string_dtype(sample_of_column)
        and sample_of_column.str.endswith((".wav", ".mp3", ".flac", ".aac", ".ogg"), na=False).all()
    ):
        return Modality.AUDIO

    return Modality.TEXT


def build_output_columns_to_generate_llm(
    df: pd.DataFrame, input_columns: list[str], nl_prompt: str
) -> tuple[list[dict[str, object]], dict[str, str]]:
    # Determine modalities of input columns
    modality_per_column = {column: _get_modality(df[column]) for column in input_columns}

    # Form prompts with examples
    samples_str, samples_image, samples_audio = _get_modality_prompts(df=df, modality_per_column=modality_per_column)
    messages = _get_feature_suggestion_message(
        nl_prompt=nl_prompt,
        input_columns=input_columns,
        samples_text=samples_str,
        samples_image=samples_image,
        samples_audio=samples_audio,
    )

    generated_output = generate_json_from_messages(messages=messages)

    generated_features: list[dict[str, object]] = []
    output_columns: dict[str, str] = {}

    # Validate that generated dicts have correct keys for later generation
    for feature in json.loads(generated_output):
        if list(feature.keys()) == ["feature_name", "feature_prompt", "input_columns"]:
            # Avoid LLMs suggesting columns that are not in input_columns
            if len(set(feature["input_columns"]) - set(df.columns)) != 0:
                feature["input_columns"] = input_columns

            generated_features.append(feature)
            output_columns[feature["feature_name"]] = feature["feature_prompt"]
        else:
            logger.info("Unable to parse some of the suggested features.")

    return generated_features, output_columns
