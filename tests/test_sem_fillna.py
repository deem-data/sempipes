import skrub

import sempipes  # pylint: disable=cyclic-import


def test_sem_fillna():
    sempipes.update_config(
        llm_for_batch_processing=sempipes.LLM(
            name="ollama/gemma3:4b",
            parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
        ),
        batch_size_for_batch_processing=5,
    )

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

    print(salaries["department"])
    print(salaries_imputed["department"])

    assert num_non_filled == 0
    assert num_mismatches < 10
