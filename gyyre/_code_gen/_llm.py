from typing import Optional
from litellm import completion, batch_completion

_DEFAULT_MODEL = "openai/gpt-4.1"


def _unwrap_python(text: str) -> str:
    prefix = "```python"
    suffix = "```"
    suffix2 = "```end"
    text = text.strip()
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(suffix):
        text = text[: -len(suffix)]
    if text.endswith(suffix2):
        text = text[: -len(suffix2)]
    text = text.strip()
    # remove lines starting with ``` or ```python (ignoring leading spaces)
    lines = text.splitlines(keepends=True)
    keep = [ln for ln in lines if not ln.lstrip().startswith("```")]
    return "".join(keep)


def _generate_result_from_messages(messages: list[dict], generate_code: bool) -> str:
    print(f"\t> Querying '{_DEFAULT_MODEL}' with {len(messages)} messages...'")

    response = completion(
        model=_DEFAULT_MODEL,
        messages=messages,
    )

    # TODO add proper error handling
    raw_code = response.choices[0].message["content"]
    if generate_code:
        return _unwrap_python(raw_code)
    else:
        return raw_code


def _get_messages(prompt: str, generate_code: bool) -> list[dict]:
    if generate_code:
        system_content = "You are a helpful assistant, assisting data scientists with completing and improving their machine learning and data preparation code."
    else:
        system_content = "You are a helpful assistant, assisting data scientists with data cleaning and preparation, for instance, completing and imputing their data."

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


def _generate_result_from_prompt(prompt: str, generate_code: bool):
    messages = _get_messages(prompt, generate_code=generate_code)
    return _generate_result_from_messages(messages, generate_code=generate_code)


def _batch_generate_results_from_prompts(
    prompts: list[str],
    generate_code: bool,
    batch_size: int = None,
    extra_args: Optional[dict] = {},
) -> list[str]:
    """
    Calls litellm.batch_completion with one message-list per prompt.
    To avoid provider rate limits, processes in batch_size.
    extra_args are additional arguments for the LLM, e.g. {"temperature": 0, "max_tokens": 64}.
    Returns a list of raw strings (or unwrapped code) aligned with `prompts`.
    """
    all_messages = [
        _get_messages(prompt=p, generate_code=generate_code) for p in prompts
    ]
    outputs = [None] * len(prompts)
    batch_size = len(all_messages) if batch_size is None else batch_size

    for start in range(0, len(all_messages), batch_size):
        sub_messages = all_messages[start : start + batch_size]

        responses = batch_completion(
            model=_DEFAULT_MODEL,
            messages=sub_messages,
            **extra_args,
        )

        for j, resp in enumerate(responses):
            try:
                content = resp.choices[0].message["content"]
                outputs[start + j] = (
                    _unwrap_python(content) if generate_code else content
                )
            except Exception:
                raise RuntimeError(f"Error processing response: {resp}")

    return outputs
