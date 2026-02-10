import skrub

import sempipes  # pylint: disable=unused-import
from sempipes.config import ensure_default_config


def test_sem_fillna():
    ensure_default_config()

    # Fetch a dataset
    dataset = skrub.datasets.fetch_employee_salaries(split="train")
    salaries = dataset.employee_salaries.head(n=60)

    # Introduce some missing values
    target_column = "department"
    salaries_dirty = salaries.copy()
    salaries_dirty.loc[30:60, target_column] = None

    salaries_dirty_ref = skrub.var("employee_salaries", salaries_dirty)

    salaries_imputed = salaries_dirty_ref.sem_fillna(
        target_column=target_column,
        nl_prompt=f"Infer the {target_column} from relevant related attributes",
        impute_with_existing_values_only=True,
        with_llm_only=True,
    ).skb.eval()

    num_mismatches = (salaries_imputed["department"] != salaries["department"]).sum()
    num_non_filled = salaries_imputed["department"].isna().sum()

    assert num_non_filled == 0
    assert num_mismatches < 10


def test_sem_fillna_baskets_fraud():
    ensure_default_config()

    dataset = skrub.datasets.fetch_credit_fraud()
    products = skrub.var("products", dataset.products)

    products = products.sem_fillna(
        target_column="make",
        nl_prompt="Infer the manufacturer from relevant product-related attributes like title or description.",
        impute_with_existing_values_only=True,
    )

    products = products.skb.eval()
    num_non_filled = products["make"].isna().sum()

    assert num_non_filled == 0
