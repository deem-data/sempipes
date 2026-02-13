import warnings

import pandas as pd

import sempipes
from experiments.house_prices_advanced_regression_techniques._sempipes_impl import sempipes_pipeline
from sempipes.optimisers import optimise_colopro
from sempipes.optimisers.montecarlo_tree_search import MonteCarloTreeSearch

warnings.filterwarnings("ignore")

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 2.0},
    ),
)

pipeline = sempipes_pipeline()

validation_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/validation.csv")

outcomes = optimise_colopro(
    pipeline,
    "house_features",
    search=MonteCarloTreeSearch(c=0.5),
    num_trials=24,
    scoring="neg_root_mean_squared_error",
    cv=5,
    additional_env_variables={"data": validation_data},
    n_jobs_for_evaluation=1,
)

best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
