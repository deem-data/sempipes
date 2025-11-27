import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from skrub import TableVectorizer
from skrub._data_ops._evaluation import find_node_by_name

from sempipes.config import ensure_default_config
from sempipes.optimisers import EvolutionarySearch, optimise_colopro


def _pipeline(X, y, operator_name):
    """Create a pipeline with cleaning operator.

    Cleaning is only applied to train data. cross_validate will handle train/test splitting internally.
    """
    salaries_info = skrub.var("salaries_info", X)

    salaries_info_cleaned = salaries_info.sem_clean(
        nl_prompt="Clean the salaries information data to improve salary prediction. Strip whitespace from text columns, normalize categorical values, impute missing values, fix outliers, and ensure numeric columns are properly formatted. Use advanced imputation techniques.",
        columns=["department", "assignment_category", "employee_position_title"],
        name=operator_name,
    )

    salaries_info_X_cleaned = salaries_info_cleaned.skb.mark_as_X()
    salaries_y = skrub.var("salaries", y).skb.mark_as_y()

    # Create pipeline: train on cleaned data
    encoded = salaries_info_X_cleaned.skb.apply(TableVectorizer())
    return encoded.skb.apply(HistGradientBoostingRegressor(random_state=0), y=salaries_y)


def _create_env(X, y, operator_name, state):
    """Create environment dictionary for learner."""
    return {
        "_skrub_X": X,
        "_skrub_y": y,
        f"sempipes_memory__{operator_name}": None,
        f"sempipes_pipeline_summary__{operator_name}": None,
        f"sempipes_prefitted_state__{operator_name}": state,
    }


def test_sem_clean_optimizable():
    ensure_default_config()

    # Fetch both train and test splits
    salaries_df_train = skrub.datasets.fetch_employee_salaries(split="train").employee_salaries
    X_train = salaries_df_train.drop(columns="current_annual_salary", errors="ignore")
    y_train = salaries_df_train["current_annual_salary"]

    salaries_df_test = skrub.datasets.fetch_employee_salaries(split="test").employee_salaries
    X_test = salaries_df_test.drop(columns="current_annual_salary", errors="ignore")
    y_test = salaries_df_test["current_annual_salary"]

    operator_name = "salary_cleaning"

    # Baseline w/o optimization
    baseline_pipeline = _pipeline(X_train, y_train, operator_name)
    data_op = find_node_by_name(baseline_pipeline, operator_name)
    empty_state = data_op._skrub_impl.estimator.empty_state()

    learner = baseline_pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(_create_env(X_train, y_train, operator_name, empty_state))
    before_optimization = learner.score(_create_env(X_test, y_test, operator_name, empty_state))

    # Optimized w/ optimization
    pipeline_to_optimise = _pipeline(X_train, y_train, operator_name)
    outcomes = optimise_colopro(
        pipeline_to_optimise,
        operator_name,
        num_trials=3,
        scoring="r2",
        search=EvolutionarySearch(population_size=2),
        cv=2,
        num_hpo_iterations_per_trial=1,
        pipeline_definition=_pipeline,
    )

    best_outcome = max(outcomes, key=lambda x: x.score)
    print(f"Best outcome score after optimization on train CV: {best_outcome.score}, state: {best_outcome.state}")

    learner_optimized = pipeline_to_optimise.skb.make_learner(fitted=False, keep_subsampling=False)
    learner_optimized.fit(_create_env(X_train, y_train, operator_name, best_outcome.state))
    after_optimization = learner_optimized.score(_create_env(X_test, y_test, operator_name, best_outcome.state))

    print(f"Before optimization: {before_optimization}, after optimization: {after_optimization}")
    assert before_optimization <= after_optimization
