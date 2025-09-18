import gyyre
from comparisons.tmdb_box_office_prediction._gyyre_impl import gyyre_pipeline

gyyre.set_config(
    gyyre.Config(
        llm_for_code_generation="openai/gpt-5-mini",
        llm_settings_for_code_generation={},
        llm_for_batch_processing="ollama/gpt-oss:20b",
        llm_settings_for_batch_processing={"api_base": "http://localhost:11434", "temperature": 0.0},
    )
)

predictions = gyyre_pipeline("comparisons/tmdb_box_office_prediction/data.csv")

memory, states = gyyre.greedy_optimise_semantic_operator(
    predictions,
    "additional_movie_features",
    num_iterations=5,
    scoring="neg_root_mean_squared_log_error",
    cv=5,
)

lowest_score = -100000000.0
corresponding_state: dict = {}

for memory_line, state in zip(memory, states):
    if memory_line["score"] > lowest_score:
        lowest_score = memory_line["score"]
        corresponding_state = state

print(f"Lowest score: {lowest_score}\n\n")
print("\n".join(corresponding_state["generated_code"]))
