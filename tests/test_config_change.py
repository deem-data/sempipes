import skrub

import sempipes


def test_different_llms():
    sempipes.set_config(
        sempipes.Config(
            llm_for_code_generation="ollama/gpt-oss:20b",
            llm_settings_for_code_generation={"api_base": "http://localhost:11434", "temperature": 0.5},
            llm_for_batch_processing="ollama/gemma3:1b",
            llm_settings_for_batch_processing={"api_base": "http://localhost:11434", "temperature": 0.0},
        )
    )

    dataset = skrub.datasets.fetch_employee_salaries(split="train")
    salaries = dataset.employee_salaries.head(n=100)

    # Introduce some missing values
    target_column = "department"
    salaries_dirty = salaries.copy()
    salaries_dirty.loc[50:100, target_column] = None

    salaries_dirty_ref = skrub.var("employee_salaries", salaries_dirty)

    _salaries_imputed = salaries_dirty_ref.sem_fillna(
        target_column=target_column,
        nl_prompt=f"Infer the {target_column} from relevant related attributes",
        impute_with_existing_values_only=True,
        with_llm_only=True,
    ).skb.eval()

    _salaries_with_features = salaries_dirty_ref.with_sem_features(
        nl_prompt="Generate additional salary features",
        name="salary_features",
        how_many=5,
    ).skb.eval()
