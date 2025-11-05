import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import TableVectorizer
from skrub._data_ops._evaluation import find_node_by_name

from experiments.colopro import Setup, TestPipeline
from sempipes.optimisers.colopro import optimise_colopro


class FraudBasketsPipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "fraudbaskets"

    def baseline(self) -> float:
        dataset = skrub.datasets.fetch_credit_fraud()

        all_baskets = dataset["baskets"]
        nonfraudulent_baskets = all_baskets[all_baskets.fraud_flag == 0]
        fraudulent_baskets = all_baskets[all_baskets.fraud_flag == 1]

        eval_baskets = pd.concat([nonfraudulent_baskets.iloc[400:1200], fraudulent_baskets.iloc[100:300]])

        operator_name = "basket_features"

        pipeline = _pipeline(eval_baskets, dataset["products"])
        data_op = find_node_by_name(pipeline, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{operator_name}"] = empty_state

        return skrub.cross_validate(learner, env, scoring="f1")["test_score"].mean()

    def optimize(self, setup: Setup) -> float:
        dataset = skrub.datasets.fetch_credit_fraud()

        all_baskets = dataset["baskets"]
        nonfraudulent_baskets = all_baskets[all_baskets.fraud_flag == 0]
        fraudulent_baskets = all_baskets[all_baskets.fraud_flag == 1]

        val_baskets = pd.concat([nonfraudulent_baskets.iloc[:400], fraudulent_baskets.iloc[:100]])
        eval_baskets = pd.concat([nonfraudulent_baskets.iloc[400:1200], fraudulent_baskets.iloc[100:300]])

        operator_name = "basket_features"

        pipeline_to_optimise = _pipeline(val_baskets, dataset["products"])
        outcomes = optimise_colopro(
            pipeline_to_optimise,
            operator_name,
            num_trials=setup.num_trials,
            scoring="f1",
            search=setup.search,
            cv=5,
            pipeline_definition=_pipeline,
            run_name=self.name,
        )

        best_outcome = max(outcomes, key=lambda x: x.score)
        state = best_outcome.state

        pipeline = _pipeline(eval_baskets, dataset["products"])
        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{operator_name}"] = state

        return skrub.cross_validate(learner, env, scoring="f1")["test_score"].mean()


def _pipeline(flagged_baskets, basket_products) -> skrub.DataOp:
    products = skrub.var("products", basket_products)
    baskets = skrub.var("baskets", flagged_baskets[["ID"]])
    labels = skrub.var("fraud_flags", flagged_baskets["fraud_flag"])

    baskets = baskets.skb.mark_as_X().skb.set_description("Potentially fraudulent shopping baskets of products")
    labels = labels.skb.mark_as_y().skb.set_description(
        "Flag indicating whether the basket is fraudulent (1) or not (0)"
    )

    aggregated = baskets.sem_agg_features(
        products,
        left_on="ID",
        right_on="basket_ID",
        nl_prompt="""
            Generate product features that are indicative of potentially fraudulent baskets, make it easy to distinguish
            anomalous baskets from regular ones! It might be helpful to combine different product statistics.
            """,
        name="basket_features",
        how_many=10,
    )

    encoded = aggregated.skb.apply(TableVectorizer())
    return encoded.skb.apply(HistGradientBoostingClassifier(), y=labels)
