import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import TableVectorizer
from skrub._data_ops._evaluation import find_node_by_name

from experiments.colopro import Setup, TestPipeline
from sempipes.optimisers.colopro import optimise_colopro


class MidwestSurveyPipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "midwestsurvey"

    def baseline(self) -> float:
        dataset = skrub.datasets.fetch_midwest_survey()

        X_eval = dataset.X.iloc[500:1500]
        X_description = dataset.metadata["description"]

        y_eval = dataset.y.iloc[500:1500]
        y_description = dataset.metadata["target"]

        operator_name = "demographic_features"

        pipeline = _pipeline(X_eval, X_description, y_eval, y_description)

        data_op = find_node_by_name(pipeline, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{operator_name}"] = empty_state

        return skrub.cross_validate(learner, env)["test_score"].mean()

    def optimize(self, setup: Setup) -> float:
        dataset = skrub.datasets.fetch_midwest_survey()

        X_val = dataset.X.iloc[0:500]
        X_eval = dataset.X.iloc[500:1500]
        X_description = dataset.metadata["description"]

        y_val = dataset.y.iloc[0:500]
        y_eval = dataset.y.iloc[500:1500]
        y_description = dataset.metadata["target"]

        operator_name = "demographic_features"

        pipeline_to_optimise = _pipeline(X_val, X_description, y_val, y_description)
        outcomes = optimise_colopro(
            pipeline_to_optimise,
            operator_name,
            num_trials=setup.num_trials,
            scoring="accuracy",
            search=setup.search,
            cv=5,
            pipeline_definition=_pipeline,
            run_name="midwest",
        )

        best_outcome = max(outcomes, key=lambda x: x.score)
        state = best_outcome.state

        pipeline = _pipeline(X_eval, X_description, y_eval, y_description)
        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{operator_name}"] = state

        return skrub.cross_validate(learner, env)["test_score"].mean()


def _pipeline(X, X_description, y, y_description) -> skrub.DataOp:
    responses = skrub.var("response", X)
    responses = responses.skb.set_description(X_description)

    labels = skrub.var("labels", y)
    labels = labels.skb.set_name(y_description)

    responses = responses.skb.mark_as_X()
    labels = labels.skb.mark_as_y()

    responses_with_additional_features = responses.with_sem_features(
        nl_prompt="""
            Compute additional demographics-related features, use your intrinsic knowledge about the US. 
            Take into account how the identification with the country or regions of it changed over the generations.         
            Also think about how the identification differs per class and education. The midwest is generally associated 
            with "Midwestern values" — friendliness, modesty, hard work, and community-mindedness.
        """,
        name="demographic_features",
        how_many=5,
    )

    encoded_responses = responses_with_additional_features.skb.apply(TableVectorizer())
    return encoded_responses.skb.apply(HistGradientBoostingClassifier(), y=labels)
