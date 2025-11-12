import warnings

from joblib.externals.loky import get_reusable_executor

import sempipes
from experiments.colopro import Setup, TestPipeline
from experiments.colopro._boxoffice import BoxOfficePipeline
from experiments.colopro._churn import ChurnPipeline
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
    num_repetitions = 3

    setup = Setup(
        # search=TreeSearch(min_num_drafts=2),
        # search=EvolutionarySearch(population_size=6),
        search=MonteCarloTreeSearch(nodes_per_expansion=2, c=0.05),
        num_trials=64,
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            # name="openai/gpt-4.1",
            parameters={"temperature": 0.8},
        ),
    )

    pipelines = [
        MidwestSurveyPipeline(),
        # BoxOfficePipeline(),
        # HealthInsurancePipeline(),
        # FraudBasketsPipeline(),
        # ChurnPipeline(),
    ]

    results = []
    for _ in range(num_repetitions):
        for pipeline in pipelines:
            results.append(_evaluate_pipeline(pipeline, setup))

    # executor = get_reusable_executor(max_workers=degree_of_parallelism, kill_workers=True)
    # futures = [
    #    executor.submit(_evaluate_pipeline, pipeline, setup) for _ in range(num_repetitions) for pipeline in pipelines
    # ]

    # results.extend(f.result() for f in futures)

    print("#" * 120)
    for name, score in results:
        print(name, score)

    # executor.shutdown(wait=True)
