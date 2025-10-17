import json

from litellm import batch_completion, completion
from tqdm import tqdm

from sempipes.config import get_config
from sempipes.llm.utils import unwrap_json, unwrap_python

_MAX_RETRIES = 5


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


def batch_generate_json_retries(
    prompts: list,  # TODO set a proper type for this
) -> list[str | None] | list[None]:
    results: list[str | None] = [None] * len(prompts)
    indices_to_retry = list(range(len(prompts)))
    attempts = 0

    while indices_to_retry and attempts < _MAX_RETRIES:
        retry_prompts = [prompts[i] for i in indices_to_retry]
        raw_results = batch_generate_results_retries(retry_prompts)

        next_retry_indices = []
        for idx, raw_result in zip(indices_to_retry, raw_results):
            try:
                if raw_result is not None:
                    unwrapped_result = unwrap_json(raw_result)
                    results[idx] = json.dumps(json.loads(unwrapped_result))
                else:
                    results[idx] = None

            except Exception as e:  # pylint: disable=broad-except
                print(f"\t> An error occurred in attempt {attempts+1} of prompt {prompts[idx]}:", e)
                # Add error to prompt for retry
                error_prompt = prompts[idx]

                error_prompt += [
                    {"role": "assistant", "content": unwrapped_result},
                    {
                        "role": "user",
                        "content": f"Parsing JSON failed with error: {type(e)} {e}.\n "
                        f"JSON: ```json{unwrapped_result}```\n Regenerate and fix the errors.",
                    },
                ]

                prompts[idx] = error_prompt
                next_retry_indices.append(idx)

        indices_to_retry = next_retry_indices
        attempts += 1

    return results


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


def batch_generate_results_retries(
    prompts: list[str] | list[dict[str, str]],
) -> list[str | None]:
    """
    Calls litellm.batch_completion with one message-list per prompt.
    Returns a list of raw strings aligned with `prompts`, with retries on failure.
    """
    results: list[str | None] = [None] * len(prompts)
    to_retry_indices = list(range(len(prompts)))
    attempts = 0

    config = get_config()
    batch_llm = config.llm_for_batch_processing
    batch_size = config.batch_size_for_batch_processing

    while to_retry_indices and attempts < _MAX_RETRIES:
        print(
            f"\t> Querying '{batch_llm.name}' with {len(to_retry_indices)} requests in batches of size {batch_size} (attempt {attempts+1})...'"
        )
        retry_prompts = [prompts[i] for i in to_retry_indices]
        batch_outputs = []

        # Batch processing
        for start_index in tqdm(range(0, len(retry_prompts), batch_size)):
            message_batch = retry_prompts[start_index : start_index + batch_size]
            try:
                responses = batch_completion(
                    model=batch_llm.name,
                    messages=message_batch,
                    **batch_llm.parameters,
                )
                for response in responses:
                    try:
                        content = response.choices[0].message["content"]
                        batch_outputs.append(content)
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Error processing response: {response}. Error: {e}")
                        batch_outputs.append(None)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Batch completion failed: {e}")
                batch_outputs.extend([None] * len(message_batch))

        next_retry_indices = []
        for idx, raw_result in zip(to_retry_indices, batch_outputs):
            if raw_result is not None:
                results[idx] = raw_result
            else:
                next_retry_indices.append(idx)

        to_retry_indices = next_retry_indices
        attempts += 1

    return results
