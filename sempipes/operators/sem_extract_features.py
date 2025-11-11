import base64
import json
import mimetypes
from enum import Enum
from random import randint
from typing import Any

import pandas as pd
import skrub
import torch
import transformers
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.llm.llm import (
    batch_generate_json_retries,
    generate_json_from_messages,
    generate_python_code_from_messages,
    get_generic_message,
)
from sempipes.operators.operators import (
    ContextAwareMixin,
    EstimatorTransformer,
    OptimisableMixin,
    SemExtractFeaturesOperator,
)

_MAX_RETRIES = 5
_SYSTEM_PROMPT = "You are an expert data scientist, assisting feature extraction from the multi-modal data such as text/images/audio."


# class syntax
class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


def _create_file_url(local_path: str) -> tuple[str, str]:
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
            samples_audio[column] = [_create_file_url(audio_sample) for audio_sample in samples_list]

        elif modality == Modality.IMAGE:
            samples_image[column] = [_create_file_url(image_sample) for image_sample in samples_list]

    return samples_str, samples_image, samples_audio


def _get_feature_suggestion_message(
    nl_prompt: str, input_columns: list[str], samples_text: str, samples_image: dict, samples_audio: dict
) -> list[dict]:
    # TODO add column description from scrub

    task_prompt = f"""
    Your goal is to help a data scientist select which features can be generated from multi-modal data from a pandas dataframe for a machine learning script which they are developing. 
    You can make your decision for each column individually or for combinations of multiple columns. 

    Please to take special care of the following: {nl_prompt}

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

    ANSWER ONLY WITH A JSON OBJECT LIKE IN THE EXAMPLE.

    MAKE SURE THAT THE NEW FEATURES HAVE MEANINGFUL NAMES.
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
        content.append({"type": "text", "text": f"Samples of image column `{column}`:"})
        for audio, audio_format in audios:
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

    Answer with only the JSON, without any additional text, explanations, comments, or formatting.
    
    Example response:
    ```json
    {example_result_str}
    ```
    """

    return task_prompt, response_example


def _get_code_feature_generation_message(
    columns_to_generate: list[str], modality_per_column: dict[str, Modality], features_to_extract: list[dict]
) -> str:
    task_prompt = f"""
Your goal is to help a data scientist generate Python code for the feature generation/extraction from multi-modal data. You need to extract information from text, image, or audio data. 
You can use any models from `transformers` library that can be used zero-shot, without additional fine-tuning. You are allowed to leverage modality-specific models for each modality.

You are provided the name of the features to extract, how to extract them, and from which columns within a pandas DataFrame `df`.

Generate Python code with method `extract_features(df: pd.DataFrame, features_to_extract: list[dict[str, object]]) -> pd.DataFrame` for feature extraction using `transformers`, `torch` libraries. 
`extract_features` takes original DataFrame `df` and a list with features to generate `features_to_extract` where each entry is a dictionary with name of the feature to generate 'feature_name', how to generate the feature 'feature_prompt', and which columns to use 'input_columns'.
`extract_features` returns original dataframe with newly generated columns: {columns_to_generate}.

Data types of the columns that should be used for the feature generation: {json.dumps(modality_per_column, indent=2)}

In the code, try to use the least loaded GPU if multiple GPUs are available.
Please use transformers=={transformers.__version__} and torch=={torch.__version__}.

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

def pick_gpu_by_free_mem_torch():
    assert torch.cuda.is_available(), "No CUDA device"
    free = []
    for i in range(torch.cuda.device_count()):
        # returns (free, total) in bytes for that device
        f, t = torch.cuda.mem_get_info(i)
        free.append((f, i))
    free.sort(reverse=True)
    _, idx = free[0]
    print(f"Chosen GPU: {{idx}}")
    return idx

if torch.cuda.device_count() > 1:
    gpu_idx = pick_gpu_by_free_mem_torch()
    device = torch.device(f"cuda:{{gpu_idx}}") # Use the selected GPU for model inference
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


features_to_extract = {features_to_extract}

def extract_features(df: pd.DataFrame, features_to_extract: list[dict[str, object]]) -> pd.DataFrame:
    # Extract features using transformers and other libraries

    ...

    # Add features to the original df as new columns
    return df
```end

"""
    post_meassage = """
The returned DataFrame `df` should contain the following columns: {columns_to_generate}.
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""
    return task_prompt + code_example + post_meassage


def _try_to_execute(
    df: pd.DataFrame, code_to_execute: str, generated_columns: list[str], features_to_extract: list[dict[str, object]]
) -> None:
    df_sample = df.head(50).copy(deep=True)

    feature_extraction_func = safe_exec(code_to_execute, "extract_features")
    extracted_sample = feature_extraction_func(df_sample, features_to_extract)

    expected_columns = generated_columns + list(df.columns)

    assert all(
        expected_column in extracted_sample.columns for expected_column in expected_columns
    ), f"The returned DataFrame does not contain all required columns. Expected: {expected_columns}. Actual: {list(extracted_sample.columns)} "

    print("\t> Code executed successfully on a sample dataframe.")


def _get_modality(column: pd.Series) -> Modality:
    sample_of_column = column.sample(frac=0.2)
    modality = None
    if (
        pd.api.types.is_string_dtype(sample_of_column)
        and sample_of_column.str.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"), na=False).all()
    ):
        modality = Modality.IMAGE
    elif (
        pd.api.types.is_string_dtype(sample_of_column)
        and sample_of_column.str.endswith(("wav", "mp3", "flac", "aac", "ogg"), na=False).all()
    ):
        modality = Modality.AUDIO
    else:
        modality = Modality.TEXT
    return modality


def _add_memorized_history(
    memory: list[dict[str, Any]] | None,
    messages: list[dict[str, str]],
    output_columns: dict[str, str],
    generated_code: list[str],
    target_metric: str,
) -> None:
    if memory is not None and len(memory) > 0:
        current_score = None

        for memory_line in memory:
            memorized_code = memory_line["update"]
            memorized_score = memory_line["score"]

            if current_score is None:
                improvement = abs(memorized_score)
            else:
                improvement = memorized_score - current_score

            if improvement > 0.0:
                generated_code.append(memorized_code)
                add_feature_sentence = (
                    "The code was executed and improved the downstream performance. "
                    "You may reuse this feature extraction code and transformers model for the next version of the code."
                )
                current_score = memorized_score
            else:
                add_feature_sentence = (
                    f"The last code changes were discarded. Try to change model, approach, or even extract different features and corresponding prompts."
                    f"(Improvement: {improvement})"
                )

            output_columns_sentence = ""
            if output_columns is not None:
                output_columns_sentence = f"Features, prompts, and columns that were used to generate this feature extraction code: {output_columns}."

            messages += [
                {"role": "assistant", "content": memorized_code},
                {
                    "role": "user",
                    "content": f"Performance for last code block: {target_metric}={memorized_score:.5f}. "
                    f"{add_feature_sentence} {output_columns_sentence} \nNext codeblock:\n",
                },
            ]


# pylint: disable=too-many-ancestors, too-many-instance-attributes
class LLMFeatureExtractor(EstimatorTransformer, TransformerMixin, ContextAwareMixin, OptimisableMixin):
    def __init__(
        self,
        nl_prompt: str,
        input_columns: list[str],
        output_columns: dict[str, str] | None,
        generate_via_code: bool,
        eval_mode: str = skrub.eval_mode(),
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
    ) -> None:
        self.nl_prompt = nl_prompt
        self.input_columns = input_columns
        self.output_columns_not_given = output_columns is None
        self.output_columns = {} if output_columns is None else output_columns
        self.generate_via_code = generate_via_code
        self.modality_per_column: dict[str, Modality] = {}
        self.generated_features_: list[dict[str, object]] = []

        self.eval_mode = eval_mode
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory

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

        generated_output = generate_json_from_messages(messages=messages)

        # Validate that generated dicts have correct keys for later generation
        for feature in json.loads(generated_output):
            if list(feature.keys()) == ["feature_name", "feature_prompt", "input_columns"]:
                # Avoid LLMs suggesting columns that are not in input_columns
                if len(set(feature["input_columns"]) - set(df.columns)) != 0:
                    feature["input_columns"] = self.input_columns

                self.generated_features_.append(feature)
                self.output_columns[feature["feature_name"]] = feature["feature_prompt"]
            else:
                print("\t> Unable to parse some of the suggested features.")

    def empty_state(self):
        state = {}
        if self.output_columns_not_given:
            state["generated_features"] = [
                {"feature_name": "", "feature_prompt": self.nl_prompt, "input_columns": self.input_columns}
            ]

        if self.generate_via_code:
            state["generated_code"] = [
                """  
def extract_features(df: pd.DataFrame, features_to_extract: list[dict[str, object]]) -> pd.DataFrame:
    return df    
"""
            ]

        return state

    def state_after_fit(self):
        state = {}
        if self.output_columns_not_given:
            state["generated_features"] = self.output_columns

        if self.generate_via_code:
            state["generated_code"] = self.generated_code_
        return state

    def memory_update_from_latest_fit(self):
        if self.generate_via_code:
            if self.generated_code_ is not None and len(self.generated_code_) > 0:
                return f"{self.generated_code_[-1]}"
        return OptimisableMixin.EMPTY_MEMORY_UPDATE

    def fit(self, df: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        if self._prefitted_state is not None:
            print(f"--- Using provided state for sempipes.sem_extract_features('{self.nl_prompt[:20]}...')")
            if self.output_columns_not_given and "generated_features" in self._prefitted_state:
                self.generated_features_ = self._prefitted_state["generated_features"]
            if self.generate_via_code:
                self.generated_code_ = self._prefitted_state["generated_code"]
            return self

        # Determine modalities of input columns
        self.modality_per_column = {column: _get_modality(df[column]) for column in self.input_columns}

        if self.output_columns == {}:
            # Ask LLM which features to generate given input columns
            print(f"--- sempipes.sem_extract_features('{self.input_columns}', '{self.nl_prompt}')")

            self._build_output_columns_to_generate_llm(df)

        else:
            # Use user-given columns
            print(
                f"--- sempipes.sem_extract_features('{self.input_columns}', '{self.output_columns}', '{self.nl_prompt}')"
            )

            for new_feature, prompt in self.output_columns.items():
                self.generated_features_.append(
                    {"feature_name": new_feature, "feature_prompt": prompt, "input_columns": self.input_columns}
                )

        print(f"\t> Generated possible columns: { self.generated_features_}")

        if self.generate_via_code:  # Extract actual features via code generation
            self.extract_features_with_code(df)

        return self

    def extract_features_with_code(self, df):
        target_metric = "accuracy"
        if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
            target_metric = self._pipeline_summary.target_metric

        # Construct prompts with multi-modal data
        prompt = _get_code_feature_generation_message(
            columns_to_generate=list(self.output_columns.keys()),
            modality_per_column=self.modality_per_column,
            features_to_extract=self.generated_features_,
        )

        # Generate code for multi-modal data
        messages = []
        generated_code = []
        for attempt in range(1, _MAX_RETRIES + 1):
            code = ""

            try:
                if attempt == 1:
                    messages += [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                    _add_memorized_history(
                        self._memory,
                        messages,
                        self.output_columns if self.output_columns_not_given else None,
                        generated_code,
                        target_metric,
                    )

                code = generate_python_code_from_messages(messages)
                code_to_execute = "\n".join(generated_code)
                code_to_execute += "\n\n" + code

                # Try to extract actual features
                _try_to_execute(
                    df,
                    code_to_execute,
                    generated_columns=list(self.output_columns.keys()),
                    features_to_extract=self.generated_features_,
                )

                generated_code.append(code)
                break
            except Exception as e:  # pylint: disable=broad-except
                print(f"\t> An error occurred in attempt {attempt}:", e)
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        f"Code: ```python{code}```\n Generate next feature (fixing error?):\n```python\n",
                    },
                ]

        self.generated_code_ = generated_code

    def extract_features_with_llm(self, df):
        # Construct prompts with multi-modal data
        prompts = []
        for _, row in df.iterrows():
            prompt = self._build_mm_generation_prompt(row=row)
            prompts.append(get_generic_message(_SYSTEM_PROMPT, prompt))

        encoded_results = batch_generate_json_retries(prompts)
        generated_columns = {}

        for encoded_result in encoded_results:
            parsed_result = json.loads(encoded_result)
            assert isinstance(parsed_result, dict), f"Expected a dict encoded in JSON, but got: {encoded_result}"

            try:
                for column_name, cell_value in parsed_result.items():
                    generated_columns[column_name] = (
                        [cell_value]
                        if generated_columns.get(column_name) is None
                        else generated_columns[column_name] + [cell_value]
                    )
            except Exception as e:
                print(f"Error processing response: {encoded_result}")
                raise e

        # Assign new results back
        for new_column, new_vals in generated_columns.items():
            df[new_column] = new_vals

        print(f"\t> Generated {len(generated_columns)} columns: {list(generated_columns.keys())}")

        return df

    def transform(self, df):
        check_is_fitted(self, "generated_features_")

        if self.generate_via_code:
            # Extract actual features
            code_to_execute = "\n".join(self.generated_code_)
            feature_extraction_func = safe_exec(code_to_execute, "extract_features")
            df = feature_extraction_func(df, self.generated_features_)

            print(f"\t> Generated columns: {list(df.columns)}. \n Code: {code_to_execute}")

        else:
            df = self.extract_features_with_llm(df=df)

        return df


class SemExtractFeaturesLLM(SemExtractFeaturesOperator):
    def generate_features_extractor(
        self,
        data_op: DataOp,
        name: str,
        nl_prompt: str,
        input_columns: list[str],
        output_columns: dict[str, str] | None = None,
        **kwargs,
    ):
        pipeline_summary = skrub.var(f"sempipes_pipeline_summary__{name}", None)
        prefitted_state = skrub.var(f"sempipes_prefitted_state__{name}", None)
        memory = skrub.var(f"sempipes_memory__{name}", [])
        eval_mode = skrub.eval_mode()
        return LLMFeatureExtractor(
            nl_prompt=nl_prompt,
            input_columns=input_columns,
            output_columns=output_columns,
            generate_via_code=kwargs.get("generate_via_code", False),
            eval_mode=eval_mode,
            _pipeline_summary=pipeline_summary,
            _prefitted_state=prefitted_state,
            _memory=memory,
        )


def sem_extract_features(
    self: DataOp,
    nl_prompt: str,
    input_columns: list[str],
    name: str = f"random_{randint(0, 10000)}_name",
    output_columns: dict[str, str] | None = None,
    **kwargs,
) -> DataOp:
    data_op = self

    feature_extractor = SemExtractFeaturesLLM().generate_features_extractor(
        data_op,
        name,
        nl_prompt,
        input_columns,
        output_columns,
        **kwargs,
    )
    return self.skb.apply(feature_extractor, how="no_wrap").skb.set_name(name)
