import nbformat
from nbclient import NotebookClient


def _run_notebook(path):
    print(f"Testing notebook: {path}...")
    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    client = NotebookClient(nb)
    client.execute()


def test_demo_notebook():
    _run_notebook("demo.ipynb")


def test_demo__sem_fillna_notebook():
    _run_notebook("demo__sem_fillna.ipynb")


def test_demo__sem_select_notebook():
    _run_notebook("demo__sem_select.ipynb")


def test_demo__greedy_optimise_semantic_operator_notebook():
    _run_notebook("demo__greedy_optimise_semantic_operator.ipynb")
