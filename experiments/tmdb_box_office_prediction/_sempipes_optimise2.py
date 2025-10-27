import warnings

import sempipes
from experiments.tmdb_box_office_prediction._sempipes_impl2 import sempipes_pipeline2
from sempipes.optimisers import optimise_colopro

warnings.filterwarnings("ignore")

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="openai/gpt-4.1",
        parameters={"temperature": 0.8},
    )
)

predictions = sempipes_pipeline2("experiments/tmdb_box_office_prediction/validation.csv", pipeline_seed=42)

outcomes = optimise_colopro(
    predictions,
    "additional_movie_features",
    num_trials=24,
    scoring="neg_root_mean_squared_log_error",
    cv=5,
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
