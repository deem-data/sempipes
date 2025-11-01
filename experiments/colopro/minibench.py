import concurrent.futures
import warnings

import sempipes
from experiments.colopro import Setup, TestPipeline
from experiments.colopro._boxoffice import BoxOfficePipeline
from experiments.colopro._fraudbaskets import FraudBasketsPipeline
from experiments.colopro._insurance import HealthInsurancePipeline
from experiments.colopro._midwest import MidwestSurveyPipeline
from sempipes.optimisers import MonteCarloTreeSearch

warnings.filterwarnings("ignore")


def _evaluate_pipeline(pipeline: TestPipeline, pipeline_setup: Setup) -> tuple[str, float]:
    sempipes.update_config(llm_for_code_generation=pipeline_setup.llm_for_code_generation)
    return pipeline.name, pipeline.optimize(pipeline_setup)


if __name__ == "__main__":
    degree_of_parallelism = 8
    num_repetitions = 6

    setup = Setup(
        # search=TreeSearch(min_num_drafts=2),
        # search=EvolutionarySearch(population_size=6),
        search=MonteCarloTreeSearch(nodes_per_expansion=2, c=0.05),
        num_trials=24,
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            # name="openai/gpt-4.1",
            parameters={"temperature": 0.3},
        ),
    )

    pipelines = [
        MidwestSurveyPipeline(),
        BoxOfficePipeline(),
        HealthInsurancePipeline(),
        FraudBasketsPipeline(),
    ]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=degree_of_parallelism) as executor:
        futures = [
            executor.submit(_evaluate_pipeline, pipeline, setup)
            for _ in range(num_repetitions)
            for pipeline in pipelines
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error: {repr(e)}")

    print("#" * 120)
    for name, score in results:
        print(name, score)
