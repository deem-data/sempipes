import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import TableVectorizer
from skrub._data_ops._evaluation import find_node_by_name

import sempipes
from experiments.colopro import Setup, TestPipeline
from sempipes.optimisers.colopro import optimise_colopro


class ChurnPipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "churn"

    def baseline(self) -> float:
        all_customers = pd.read_csv("experiments/colopro/churn_customers.csv")
        all_transactions = pd.read_csv("experiments/colopro/churn_transactions.csv")

        eval_customers = all_customers[500:1500]

        operator_name = "transaction_features"

        pipeline = _pipeline(eval_customers, all_transactions)
        data_op = find_node_by_name(pipeline, operator_name)
        empty_state = data_op._skrub_impl.estimator.empty_state()

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{operator_name}"] = empty_state

        return skrub.cross_validate(learner, env, scoring="accuracy")["test_score"].mean()

    def optimize(self, setup: Setup) -> float:
        all_customers = pd.read_csv("experiments/colopro/churn_customers.csv")
        all_transactions = pd.read_csv("experiments/colopro/churn_transactions.csv")

        val_customers = all_customers[:500]
        eval_customers = all_customers[500:1500]
        operator_name = "transaction_features"

        pipeline_to_optimise = _pipeline(val_customers, all_transactions)
        outcomes = optimise_colopro(
            pipeline_to_optimise,
            operator_name,
            num_trials=setup.num_trials,
            scoring="accuracy",
            search=setup.search,
            cv=5,
            pipeline_definition=_pipeline,
            run_name=self.name,
        )

        best_outcome = max(outcomes, key=lambda x: x.score)
        state = best_outcome.state

        pipeline = _pipeline(eval_customers, all_transactions)
        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{operator_name}"] = state

        return skrub.cross_validate(learner, env, scoring="accuracy")["test_score"].mean()


def _pipeline(labeled_customers, customer_transactions) -> skrub.DataOp:
    customer_ids = skrub.var("customers", labeled_customers[["CustomerID"]])
    churned = skrub.var("churned", labeled_customers["has_churned"])
    transactions = skrub.var("transactions", customer_transactions)

    customer_ids = sempipes.as_X(customer_ids, "Identifiers of customers")
    churned = sempipes.as_y(churned, "Churn status per customer")

    num_transactions = transactions.groupby("CustomerID").size().reset_index(name="num_transactions")
    customer_ids = customer_ids.merge(num_transactions, on="CustomerID", how="left")
    customer_ids = customer_ids.assign(num_transactions=lambda df: df["num_transactions"].fillna(0).astype(int))

    aggregated = customer_ids.sem_agg_features(
        transactions,
        left_on="CustomerID",
        right_on="CustomerID",
        nl_prompt="""
            Compute features from the shopping transactions that indicate whether a customer will buy something in 
            the following years or not.
            """,
        name="transaction_features",
        how_many=10,
    )

    encoded = aggregated.skb.apply(TableVectorizer())
    return encoded.skb.apply(HistGradientBoostingClassifier(), y=churned)
