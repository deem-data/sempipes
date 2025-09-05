# sempipes

This repository contains the code and resources for the `gyyre` project. The project aims to build a semantic pipeline for data processing.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Formatting](#formatting)
- [Tests](#tests)

## Features

- sem_fillna
- with_sem_features
- apply_with_sem_choose

## Installation

To install and set up the project locally, follow these steps:

```bash
git clone https://github.com/deem-data/sempipes.git
cd sempipes
poetry install
poetry run pre-commit install
```

# Formatting

Run either
```bash
poetry run black .
poetry run pylint tests/
poetry run pylint gyyre/
```

or

```bash
poetry run pre-commit run --all-files
```

# Tests

```
poetry run pytest -q
```
