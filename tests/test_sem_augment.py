import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import sempipes
from sempipes.config import ensure_default_config


def test_sem_augment_via_code():
    ensure_default_config()

    # Fetch a dataset
    salaries_df = skrub.datasets.fetch_employee_salaries(split="train").employee_salaries
    salaries = skrub.var("salaries", salaries_df)
    salaries_X = salaries.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
    salaries_y = salaries["current_annual_salary"].skb.mark_as_y()

    new_salaries_X = skrub.datasets.fetch_employee_salaries(split="test").X
    new_salaries_y = skrub.datasets.fetch_employee_salaries(split="test").y

    # Evaluate before augmentation
    annual_salary = salaries_X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=salaries_y
    )
    learner = annual_salary.skb.make_learner(fitted=True)
    predictions = learner.predict({"salaries": new_salaries_X})
    r2_before_augmentation = r2_score(new_salaries_y, predictions)
    print(f"Salaries {salaries_X.shape} R2 before augmentation: {r2_before_augmentation}")

    # Augment the dataset
    salaries_augmented = salaries.sem_augment(
        nl_prompt="Augment data to improve 'current_annual_salary' prediction. Ensure that the augmented values are realistic and within a reasonable range based on the existing data. Maintain logical consistency across related columns. Avoid creating duplicate rows and ensure that categorical variables remain valid.",
        number_of_rows_to_generate=2000,
        name="augment_salaries",
        generate_via_code=True,
    )

    salaries_X_augmented = salaries_augmented.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
    salaries_y_augmented = salaries_augmented["current_annual_salary"].skb.mark_as_y()

    # Evaluate after augmentation
    annual_salary_augmented = salaries_X_augmented.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=salaries_y_augmented
    )
    learner_augmented = annual_salary_augmented.skb.make_learner(fitted=True)
    predictions_augmented = learner_augmented.predict({"salaries": new_salaries_X})
    r2_after_augmentation = r2_score(new_salaries_y, predictions_augmented)
    print(f"Salaries {salaries_X_augmented.skb.eval().shape} R2 after augmentation: {r2_after_augmentation}")

    assert r2_before_augmentation < r2_after_augmentation


def test_sem_augment_via_data():
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="openai/gpt-4.1",
            parameters={"temperature": 0.0},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="openai/gpt-4.1",
            parameters={"temperature": 0.0},
        ),
    )

    # Fetch a dataset
    salaries_df = skrub.datasets.fetch_employee_salaries(split="train").employee_salaries
    salaries = skrub.var("salaries", salaries_df)
    salaries_X = salaries.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
    salaries_y = salaries["current_annual_salary"].skb.mark_as_y()

    new_salaries_X = skrub.datasets.fetch_employee_salaries(split="test").X
    new_salaries_y = skrub.datasets.fetch_employee_salaries(split="test").y

    # Evaluate before augmentation
    annual_salary = salaries_X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=salaries_y
    )
    learner = annual_salary.skb.make_learner(fitted=True)
    predictions = learner.predict({"salaries": new_salaries_X})
    r2_before_augmentation = r2_score(new_salaries_y, predictions)
    print(f"Salaries {salaries_X.shape} R2 before augmentation: {r2_before_augmentation}")

    # Augment the dataset
    salaries_augmented = salaries.sem_augment(
        nl_prompt="Augment data to improve 'current_annual_salary' prediction.",
        number_of_rows_to_generate=2000,
        name="augment_salaries",
        generate_via_code=False,
    )

    salaries_augmented.fillna(method="ffill", inplace=True)

    salaries_X_augmented = salaries_augmented.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
    salaries_y_augmented = salaries_augmented["current_annual_salary"].skb.mark_as_y()

    # Evaluate after augmentation
    annual_salary_augmented = salaries_X_augmented.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=salaries_y_augmented
    )
    learner_augmented = annual_salary_augmented.skb.make_learner(fitted=True)
    predictions_augmented = learner_augmented.predict({"salaries": new_salaries_X})
    r2_after_augmentation = r2_score(new_salaries_y, predictions_augmented)
    print(f"Salaries {salaries_X_augmented.skb.eval().shape} R2 after augmentation: {r2_after_augmentation}")

    assert r2_before_augmentation <= r2_after_augmentation
