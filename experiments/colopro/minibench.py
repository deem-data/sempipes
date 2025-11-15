import warnings

import numpy as np

import sempipes
from experiments.colopro import Setup, TestPipeline
from experiments.colopro._boxoffice import BoxOfficePipeline
from experiments.colopro._churn import ChurnPipeline
from experiments.colopro._fraudbaskets import FraudBasketsPipeline
from experiments.colopro._insurance import HealthInsurancePipeline
from experiments.colopro._midwest import MidwestSurveyPipeline
from sempipes.optimisers import EvolutionarySearch, MonteCarloTreeSearch, TreeSearch

warnings.filterwarnings("ignore")


def _evaluate_pipeline(pipeline: TestPipeline, seed, pipeline_setup: Setup) -> tuple[str, float]:
    sempipes.update_config(llm_for_code_generation=pipeline_setup.llm_for_code_generation)
    return pipeline.name, seed, pipeline.optimize(seed, pipeline_setup)


if __name__ == "__main__":
    seeds = [0, 42, 18102022, 1985, 748]

    model_name = "gemini/gemini-2.5-flash"
    model_temperature = 2.0
    search_name = "evo_search"

    setup = Setup(
        # search=TreeSearch(min_num_drafts=2),
        search=EvolutionarySearch(population_size=6),
        # search=MonteCarloTreeSearch(nodes_per_expansion=2, c=0.05),
        num_trials=36,
        llm_for_code_generation=sempipes.LLM(
            name=model_name,
            # name="openai/gpt-4.1",
            parameters={"temperature": model_temperature},
        ),
    )

    pipelines = [
        MidwestSurveyPipeline(),
        BoxOfficePipeline(),
        HealthInsurancePipeline(),
        FraudBasketsPipeline(),
        ChurnPipeline(),
    ]

    results = []

    import sys

    if len(sys.argv) < 2:
        available_pipeline_names = [p.name for p in pipelines]
        print(
            f"Error: You must provide a pipeline name as a command line argument. "
            f"Available pipeline names: {', '.join(available_pipeline_names)}",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline_name = sys.argv[1]
    pipeline = next(p for p in pipelines if p.name == pipeline_name)

    # Print headers and results with model name and temperature
    for seed in seeds:
        try:
            np.random.seed(seed)
            name, seed, scores_over_time = _evaluate_pipeline(pipeline, seed, setup)
            for max_trial, chosen_trial, train_score, test_score in scores_over_time:
                print(
                    f"MINIBENCH_RESULT> {model_name},{model_temperature},{search_name},{name},{seed},{max_trial},{chosen_trial},{train_score},{test_score}"
                )
                results.append((name, seed, max_trial, chosen_trial, train_score, test_score))
        except Exception as e:
            print(f"MINIBENCH_ERROR> {e} for seed {seed}")
            pass

    print("#" * 120)
    for name, seed, max_trial, chosen_trial, train_score, test_score in results:
        print(
            f"{model_name},{model_temperature},{search_name},{name},{seed},{max_trial},{chosen_trial},{train_score},{test_score}"
        )
