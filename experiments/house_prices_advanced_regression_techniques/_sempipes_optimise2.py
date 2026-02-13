import warnings

import pandas as pd

import sempipes
from experiments.house_prices_advanced_regression_techniques._sempipes_impl2 import sempipes_pipeline2
from sempipes.optimisers import optimise_colopro
from sempipes.optimisers.montecarlo_tree_search import MonteCarloTreeSearch

warnings.filterwarnings("ignore")

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 2.0},
    ),
)

data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/validation.csv")

pipeline = sempipes_pipeline2()

outcomes = optimise_colopro(
    pipeline,
    "house_features",
    num_trials=24,
    scoring="neg_root_mean_squared_log_error",
    search=MonteCarloTreeSearch(c=0.5),
    cv=5,
    additional_env_variables={"data": data},
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
