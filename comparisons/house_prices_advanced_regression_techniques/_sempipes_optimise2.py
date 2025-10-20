import warnings

import sempipes
from comparisons.house_prices_advanced_regression_techniques._sempipes_impl2 import sempipes_pipeline2
from sempipes.optimisers import optimise_olopro

warnings.filterwarnings("ignore")

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="openai/gpt-4.1",
        parameters={"temperature": 0.8},
    ),
)

pipeline = sempipes_pipeline2("comparisons/house_prices_advanced_regression_techniques/validation.csv")

outcomes = optimise_olopro(
    pipeline,
    "house_features",
    num_trials=24,
    scoring="neg_root_mean_squared_log_error",
    cv=5,
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
