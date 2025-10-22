import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import sempipes


def test_sem_distill_via_code():
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="openai/gpt-4.1",
            parameters={"temperature": 0.0},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="ollama/gpt-oss:20b",
            parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
        ),
    )
    salaries_df = skrub.datasets.fetch_employee_salaries(split="train").employee_salaries
    salaries = skrub.var("salaries", salaries_df)
    salaries_X = salaries.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
    salaries_y = salaries["current_annual_salary"].skb.mark_as_y()
    new_salaries_X = skrub.datasets.fetch_employee_salaries(split="test").X
    new_salaries_y = skrub.datasets.fetch_employee_salaries(split="test").y
    annual_salary = salaries_X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=salaries_y
    )
    learner = annual_salary.skb.make_learner(fitted=True)
    predictions = learner.predict({"salaries": new_salaries_X})
    r2_before_distill = r2_score(new_salaries_y, predictions)
    print(f"Salaries {salaries_X.shape} R2 before distill: {r2_before_distill}")
    salaries_distilled = salaries.sem_distill(
        nl_prompt="Distill data to improve 'current_annual_salary' prediction. Use advanced methods for the distillation.",
        number_of_rows=5000,
    )
    salaries_X_distilled = salaries_distilled.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
    salaries_y_distilled = salaries_distilled["current_annual_salary"].skb.mark_as_y()
    annual_salary_distilled = salaries_X_distilled.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=salaries_y_distilled
    )
    learner_distilled = annual_salary_distilled.skb.make_learner(fitted=True)
    predictions_distilled = learner_distilled.predict({"salaries": new_salaries_X})
    r2_after_distill = r2_score(new_salaries_y, predictions_distilled)
    print(f"Salaries {salaries_X_distilled.skb.eval().shape} R2 after distill: {r2_after_distill}")
    assert r2_before_distill - 0.05 <= r2_after_distill
