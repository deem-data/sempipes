from litellm import batch_completion, completion
from tqdm import tqdm

from sempipes.config import get_config
from sempipes.llm.utils import unwrap_json, unwrap_python


def get_cleaning_message(prompt: str) -> list[dict]:
    return get_generic_message(
        "You are a helpful assistant, assisting data scientists with data cleaning and preparation, for instance, completing and imputing their data.",
        prompt,
    )


def get_generic_message(system_content: str, user_content: str | list[dict]) -> list[dict]:
    return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]


def generate_python_code(prompt: str) -> str:
    messages = get_generic_message(
        "You are a helpful assistant, assisting data scientists with completing and improving their machine learning and data preparation code.",
        prompt,
    )
    return generate_python_code_from_messages(messages)


def generate_python_code_from_messages(messages: list[dict]) -> str:
    raw_code = _generate_code_from_messages(messages=messages)
    return unwrap_python(raw_code)


def _generate_code_from_messages(messages: list[dict]) -> str:
    code_gen_llm = get_config().llm_for_code_generation
    print(f"\t> Querying '{code_gen_llm.name}' with {len(messages)} messages...'")

    response = completion(
        model=code_gen_llm.name,
        messages=messages,
        **code_gen_llm.parameters,
    )

    # TODO add proper error handling
    raw_code = response.choices[0].message["content"]
    return raw_code


def generate_json_from_messages(messages: list[dict]) -> str:
    raw_code = _generate_code_from_messages(messages=messages)
    return unwrap_json(raw_code)


def batch_generate_json(
    prompts: list[str],
) -> list[str | None]:
    raw_results = batch_generate_results(prompts)
    return [unwrap_json(raw_result) if raw_result is not None else None for raw_result in raw_results]


def batch_generate_results(
    prompts: list[str],
) -> list[str | None]:
    """
    Calls litellm.batch_completion with one message-list per prompt.
    Returns a list of raw strings aligned with `prompts`.
    """

    config = get_config()
    batch_llm = config.llm_for_batch_processing
    batch_size = config.batch_size_for_batch_processing
    print(f"\t> Querying '{batch_llm.name}' with {len(prompts)} requests in batches of size {batch_size}...'")

    outputs = []

    for start_index in tqdm(range(0, len(prompts), batch_size)):
        message_batch = prompts[start_index : start_index + batch_size]

        responses = batch_completion(
            model=batch_llm.name,
            messages=message_batch,
            **batch_llm.parameters,
        )

        for response in responses:
            try:
                content = response.choices[0].message["content"]
                outputs.append(content)
            except Exception as e:
                print(f"Error processing response: {response}")
                raise e

    return outputs
