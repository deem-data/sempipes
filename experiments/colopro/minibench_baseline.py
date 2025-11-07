from experiments.colopro._boxoffice import BoxOfficePipeline
from experiments.colopro._churn import ChurnPipeline
from experiments.colopro._fraudbaskets import FraudBasketsPipeline
from experiments.colopro._insurance import HealthInsurancePipeline
from experiments.colopro._midwest import MidwestSurveyPipeline

if __name__ == "__main__":
    pipelines = [
        MidwestSurveyPipeline(),
        BoxOfficePipeline(),
        HealthInsurancePipeline(),
        FraudBasketsPipeline(),
        ChurnPipeline(),
    ]

    results = [(pipeline.name, pipeline.baseline()) for pipeline in pipelines]

    print("#" * 120)
    for name, score in results:
        print(name, score)
