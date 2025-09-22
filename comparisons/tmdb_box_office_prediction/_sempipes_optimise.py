import sempipes
from comparisons.tmdb_box_office_prediction._sempipes_impl import sempipes_pipeline

sempipes.set_config(
    sempipes.Config(
        llm_for_code_generation=sempipes.LLM(
            name="openai/gpt-5-mini",
            parameters={},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="ollama/gemma3:1b",
            parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
        ),
    )
)

predictions = sempipes_pipeline("comparisons/tmdb_box_office_prediction/validation.csv")

outcomes = sempipes.optimise_olopro(
    predictions,
    "additional_movie_features",
    budget=5,
    scoring="neg_root_mean_squared_log_error",
    cv=5,
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
