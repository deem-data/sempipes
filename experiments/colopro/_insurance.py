import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import TableVectorizer
from skrub._data_ops._evaluation import find_node_by_name

from experiments.colopro import Setup, TestPipeline
from sempipes.optimisers.colopro import optimise_colopro


class HealthInsurancePipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "insurance"

    def baseline(self) -> float:
        data = pd.read_csv("experiments/colopro/insurance.csv")

        X_eval = data.iloc[500:1500]
        operator_name = "more_features"

        pipeline = _pipeline(X_eval)
        data_op = find_node_by_name(pipeline, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{operator_name}"] = empty_state

        return skrub.cross_validate(learner, env)["test_score"].mean()

    def optimize(self, setup: Setup) -> float:
        data = pd.read_csv("experiments/colopro/insurance.csv")

        X_val = data.iloc[0:500]
        X_eval = data.iloc[500:1500]

        operator_name = "more_features"

        pipeline_to_optimise = _pipeline(X_val)
        outcomes = optimise_colopro(
            pipeline_to_optimise,
            operator_name,
            num_trials=setup.num_trials,
            scoring="accuracy",
            search=setup.search,
            cv=5,
            pipeline_definition=_pipeline,
            run_name="insurance",
        )

        best_outcome = max(outcomes, key=lambda x: x.score)
        state = best_outcome.state

        pipeline = _pipeline(X_eval)
        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{operator_name}"] = state

        return skrub.cross_validate(learner, env)["test_score"].mean()


def _pipeline(data) -> skrub.DataOp:
    records = skrub.var("records", data)
    records.skb.set_description("""
        Raw data for insurance Lead Prediction. A policy is recommended to a person when they land on an insurance website, and if the person chooses to fill up a form to apply, it is considered a Positive outcome (Classified as lead). All other conditions are considered Zero outcomes.
    """)
    labels = records["Response"]
    labels.skb.set_description("""
        Insurance policy is recommended to a person when Response is 1.
    """)
    records = records.drop("Response", axis=1)

    # labels = labels.skb.set_name(y_description)

    records = records.skb.mark_as_X()
    labels = labels.skb.mark_as_y()

    records_with_additional_features = records.with_sem_features(
        nl_prompt="""
            Compute additional features that could help predict whether a person will apply for an insurance policy based on their demographics, and other relevant data. Consider factors such as age, income, and previous insurance history to derive meaningful features that enhance the predictive power of the model.
        """,
        name="more_features",
        how_many=10,
    )

    encoded_responses = records_with_additional_features.skb.apply(TableVectorizer())
    return encoded_responses.skb.apply(HistGradientBoostingClassifier(), y=labels)
