import warnings

import sempipes
from comparisons.scrabble_player_rating._sempipes_impl import sempipes_pipeline

warnings.filterwarnings("ignore")

sempipes.set_config(
    sempipes.Config(
        llm_for_code_generation="gemini/gemini-2.5-pro",
        llm_settings_for_code_generation={"temperature": 0.0},
        llm_for_batch_processing="ollama/gpt-oss:20b",
        llm_settings_for_batch_processing={"api_base": "http://localhost:11434", "temperature": 0.0},
    )
)


pipeline = sempipes_pipeline("comparisons/scrabble_player_rating/validation.csv")

memory, states = sempipes.greedy_optimise_semantic_operator(
    pipeline,
    "player_features",
    num_iterations=10,
    scoring="neg_root_mean_squared_error",
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
