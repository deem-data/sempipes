import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

example_files = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize(
    "example",
    example_files,
    ids=[f.stem for f in example_files],
)
def test_example(example):
    module_name = f"examples.{example.stem}"
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=str(EXAMPLES_DIR.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{module_name} failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
