from litellm import completion

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
