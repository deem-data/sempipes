from litellm import batch_completion, completion

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


def _get_cleaning_message(prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant, assisting data scientists with data cleaning and preparation, for instance, completing and imputing their data.",
        },
        {"role": "user", "content": prompt},
    ]


def _generate_python_code(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant, assisting data scientists with completing and improving their machine learning and data preparation code.",
        },
        {"role": "user", "content": prompt},
    ]
    return _generate_python_code_from_messages(messages)


def _generate_python_code_from_messages(messages: list[dict]) -> str:
    print(f"\t> Querying '{_DEFAULT_MODEL}' with {len(messages)} messages...'")

    response = completion(
        model=_DEFAULT_MODEL,
        messages=messages,
        temperature=0.0,
    )

    # TODO add proper error handling
    raw_code = response.choices[0].message["content"]
    return _unwrap_python(raw_code)


def _batch_generate_results(
    prompts: list[str],
    batch_size: int,
) -> list[str | None]:
    """
    Calls litellm.batch_completion with one message-list per prompt.
    Returns a list of raw strings aligned with `prompts`.
    """
    assert batch_size is not None and batch_size > 0, "batch_size must be a positive integer"

    all_messages = [_get_cleaning_message(prompt) for prompt in prompts]
    outputs = []

    for start_index in range(0, len(prompts), batch_size):
        message_batch = all_messages[start_index : start_index + batch_size]

        responses = batch_completion(
            model=_DEFAULT_MODEL,
            messages=message_batch,
            temperature=0.0,
        )

        for response in responses:
            try:
                content = response.choices[0].message["content"]
                outputs.append(content)
            except Exception as e:
                print(f"Error processing response: {response}")
                raise e

    return outputs
