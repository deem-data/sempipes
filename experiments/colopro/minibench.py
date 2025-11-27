import sys
import warnings

import numpy as np

import sempipes
from experiments.colopro import Setup, TestPipeline
from experiments.colopro._churn import ChurnPipeline
from experiments.colopro._fraudbaskets import FraudBasketsPipeline
from experiments.colopro._midwest import MidwestSurveyPipeline
from experiments.colopro._traffic import TrafficPipeline
from sempipes.logging import get_logger
from sempipes.optimisers import EvolutionarySearch, MonteCarloTreeSearch, TreeSearch

warnings.filterwarnings("ignore")


def _evaluate_pipeline(pipeline: TestPipeline, seed, pipeline_setup: Setup) -> tuple[str, float]:
    sempipes.update_config(llm_for_code_generation=pipeline_setup.llm_for_code_generation)
    return pipeline.optimize(seed, pipeline_setup)


if __name__ == "__main__":
    logger = get_logger()

    pipelines = [
        MidwestSurveyPipeline(),
        FraudBasketsPipeline(),
        ChurnPipeline(),
        TrafficPipeline(),
    ]

    searches = {
        "tree_search": TreeSearch(min_num_drafts=2),
        "evo_search": EvolutionarySearch(population_size=6),
        "mct_search": MonteCarloTreeSearch(nodes_per_expansion=2, c=0.05),
    }

    if len(sys.argv) < 4:
        available_pipeline_names = [p.name for p in pipelines]
        print(
            f"Error: You must provide a pipeline name as a command line argument. "
            f"Available pipeline names: {', '.join(available_pipeline_names)}",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline_name = sys.argv[1]
    pipeline = next(p for p in pipelines if p.name == pipeline_name)

    if sys.argv[2] == "gpt":
        model_name = "openai/gpt-4.1"
        model_temperature = 0.8
        model_parameters = {"temperature": model_temperature}
    elif sys.argv[2] == "gemini":
        model_name = "gemini/gemini-2.5-flash"
        model_temperature = 2.0
        model_parameters = {"temperature": model_temperature}
    elif sys.argv[2] == "qwen":
        model_name = "ollama/qwen2.5-coder:32b"
        model_temperature = 0.8
        model_parameters = {"temperature": model_temperature, "api_base": "http://localhost:12000"}
    else:
        print(
            f"Error: Invalid model name: {sys.argv[2]}",
            file=sys.stderr,
        )
        sys.exit(1)

    search_name = sys.argv[3]

    if search_name not in searches:
        print(
            f"Error: Invalid search name: {search_name}",
            file=sys.stderr,
        )
        sys.exit(1)

    sempipes.update_config(prefer_empty_state_in_preview=True)

    seeds = [0, 42, 18102022, 1985, 748]

    llm = sempipes.LLM(name=model_name, parameters=model_parameters)

    setup = Setup(
        search=searches[search_name],
        num_trials=36,
        llm_for_code_generation=llm,
    )

    results = []

    logger.info(
        f"Starting mini-benchmark for pipeline {pipeline_name}  with model {model_name}, temperature {model_temperature} and search {search_name}."
    )

    for seed in seeds:
        try:
            np.random.seed(seed)
            scores_over_time = _evaluate_pipeline(pipeline, seed, setup)
            name = pipeline.name
            for max_trial, chosen_trial, train_score, test_score in scores_over_time:
                print(
                    f"MINIBENCH_RESULT> {model_name},{model_temperature},{search_name},{name},{seed},{max_trial},{chosen_trial},{train_score},{test_score}"
                )
                results.append((name, seed, max_trial, chosen_trial, train_score, test_score))
        except Exception as e:
            logger.error(f"MINIBENCH_ERROR> {e} for seed {seed}", exc_info=True)
            pass

    print("#" * 120)
    for name, seed, max_trial, chosen_trial, train_score, test_score in results:
        print(
            f"{model_name},{model_temperature},{search_name},{name},{seed},{max_trial},{chosen_trial},{train_score},{test_score}"
        )
