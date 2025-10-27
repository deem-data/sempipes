import sempipes
from experiments.tmdb_box_office_prediction._sempipes_impl import sempipes_pipeline
from sempipes.optimisers import optimise_colopro

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-2.5-pro",
        parameters={"temperature": 0.0},
    ),
)

predictions = sempipes_pipeline("experiments/tmdb_box_office_prediction/validation.csv")

outcomes = optimise_colopro(
    predictions,
    "additional_movie_features",
    num_trials=5,
    scoring="neg_root_mean_squared_log_error",
    cv=5,
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
