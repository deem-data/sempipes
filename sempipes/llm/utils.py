import re

from sempipes.logging import get_logger

logger = get_logger()


def _unwrap(text: str, prefix, suffixes) -> str:
    text = text.strip()
    if text.startswith(prefix):
        text = text[len(prefix) :]
    for suffix in suffixes:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = text.strip()
    # remove lines starting with ``` or ```python (ignoring leading spaces)
    lines = text.splitlines(keepends=True)
    keep = [ln for ln in lines if not ln.lstrip().startswith("```")]
    return "".join(keep)


def unwrap_json(text: str) -> str:
    # Check if the additional output
    match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Remove comments (// ... until end of line)
    text = re.sub(r"//.*", "", text)
    return _unwrap(text=text, prefix="```json", suffixes=["```", "```end", "\nend"])


def unwrap_python(text: str | None) -> str:
    assert text is not None, "Response text to unwrap is None"
    return _unwrap(text=text, prefix="```python", suffixes=["```", "```end", "\nend"])
