import warnings

import sempipes
from comparisons.house_prices_advanced_regression_techniques._sempipes_impl import sempipes_pipeline
from sempipes.optimisers import optimise_olopro

warnings.filterwarnings("ignore")

sempipes.set_config(
    sempipes.Config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-pro",
            parameters={"temperature": 0.0},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="ollama/gemma3:1b",
            parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
        ),
    )
)

pipeline = sempipes_pipeline("comparisons/house_prices_advanced_regression_techniques/validation.csv")

outcomes = optimise_olopro(
    pipeline,
    "house_features",
    num_trials=5,
    scoring="neg_root_mean_squared_error",
    cv=5,
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
