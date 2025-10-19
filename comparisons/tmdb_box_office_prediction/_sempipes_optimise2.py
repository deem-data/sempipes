import warnings

import sempipes
from comparisons.tmdb_box_office_prediction._sempipes_impl2 import sempipes_pipeline2
from sempipes.optimisers import optimise_olopro

warnings.filterwarnings("ignore")

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="anthropic/claude-haiku-4-5-20251001",
        parameters={"temperature": 0.0},
    )
)

predictions = sempipes_pipeline2("comparisons/tmdb_box_office_prediction/validation.csv", pipeline_seed=42)

outcomes = optimise_olopro(
    predictions,
    "additional_movie_features",
    num_trials=24,
    scoring="neg_root_mean_squared_log_error",
    cv=5,
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
