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
    return _unwrap(text=text, prefix="```json", suffix="```", suffix2="```end")


def _unwrap_python(text: str) -> str:
    return _unwrap(text=text, prefix="```python", suffix="```", suffix2="```end")
