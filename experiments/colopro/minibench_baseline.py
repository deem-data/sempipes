from experiments.colopro._boxoffice import BoxOfficePipeline
from experiments.colopro._insurance import HealthInsurancePipeline
from experiments.colopro._midwest import MidwestSurveyPipeline

if __name__ == "__main__":
    pipelines = [
        MidwestSurveyPipeline(),
        BoxOfficePipeline(),
        HealthInsurancePipeline(),
    ]

    results = [(pipeline.name, pipeline.baseline()) for pipeline in pipelines]

    print("#" * 120)
    for name, score in results:
        print(name, score)
