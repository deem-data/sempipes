import skrub
from gyyre import *


def test_sem_fillna():
    # Fetch a dataset
    dataset = skrub.datasets.fetch_employee_salaries(split="train")
    employee_salaries = dataset.employee_salaries

    # Introduce some missing values
    target_column = "department"
    ix_na = employee_salaries.sample(frac=0.05).index
    employee_salaries_dirty = employee_salaries.copy()
    employee_salaries_dirty.loc[ix_na, target_column] = None

    employee_salaries_dirty = skrub.var("employee_salaries", employee_salaries_dirty)

    employee_salaries_dirty = employee_salaries_dirty.sem_fillna(
        target_column=target_column,
        nl_prompt=f"Infer the {target_column} from relevant related attributes.",
    )

    res = (
        (
            employee_salaries_dirty.loc[ix_na, "department"]
            != employee_salaries.loc[ix_na, "department"]
        )
        .sum()
        .skb.eval()
    )

    assert (
        res == 0
        and employee_salaries_dirty.loc[ix_na, "department"].isna().sum().skb.eval()
        == 0
    )
