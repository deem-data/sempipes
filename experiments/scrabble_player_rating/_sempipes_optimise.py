import warnings

import sempipes
from experiments.scrabble_player_rating._sempipes_impl import sempipes_pipeline
from sempipes.optimisers import optimise_colopro

warnings.filterwarnings("ignore")

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="openai/gpt-4.1",
        parameters={"temperature": 0.8},
    ),
)


pipeline = sempipes_pipeline("experiments/scrabble_player_rating/validation.csv")

outcomes = optimise_colopro(
    pipeline,
    "player_features",
    num_trials=24,
    scoring="neg_root_mean_squared_error",
    cv=5,
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
print("\n".join(best_outcome.state["generated_code"]))
