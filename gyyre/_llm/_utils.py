import re


def _unwrap(text: str, prefix, suffix, suffix2) -> str:
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


def _unwrap_json(text: str) -> str:
    # Check if the additional output
    match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Remove comments (// ... until end of line)
    text = re.sub(r"//.*", "", text)
    
    return _unwrap(text=text, prefix="```json", suffix="```", suffix2="```end")


def _unwrap_python(text: str) -> str:
    # Check for the additional output
    match = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("Error parsing Python code.")

    text = match.group(1).strip()

    return _unwrap(text=text, prefix="```python", suffix="```", suffix2="```end")
