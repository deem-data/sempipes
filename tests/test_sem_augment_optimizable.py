import skrub
from sklearn.ensemble import GradientBoostingRegressor
from skrub import TableVectorizer
from skrub._data_ops._evaluation import find_node_by_name

from sempipes.config import ensure_default_config
from sempipes.optimisers import EvolutionarySearch, optimise_colopro


def _pipeline(salaries_df, operator_name):
    """Create a pipeline with augmentation operator.

    Augmentation is only applied to train data. cross_validate will handle train/test splitting internally.
    """
    salaries = skrub.var("salaries", salaries_df)

    # Augment the train data
    salaries_augmented = salaries.sem_augment(
        nl_prompt="Augment data to improve 'current_annual_salary' prediction with sklearn's GradientBoostingRegressor. First, analyze the distribution of features and target values in df to identify underrepresented regions in feature space (e.g., rare combinations of job titles, departments, and experience levels). Generate new rows that fill these gaps while maintaining realistic relationships: (1) Ensure salary values are consistent with job titles, departments, and years of experience based on observed patterns in the data, (2) Maintain logical consistency across related columns (e.g., department-job title combinations should make sense), (3) Use statistical sampling (e.g., from observed distributions or ranges) rather than random values, (4) Prioritize generating examples in underrepresented feature combinations that could help the model learn better decision boundaries, (5) Ensure all augmented values are within reasonable ranges observed in the data (check min/max for numeric columns, valid categories for categorical columns), (6) Avoid creating duplicate rows by checking for exact matches before appending. Use pandas operations, numpy for sampling, and maintain all original columns. Return df with appended rows. NO RECURSIVE OT OVERCOMPLICATED CODE.",
        number_of_rows_to_generate=1000,
        generate_via_code=True,
        name=operator_name,
    )

    # Extract X and y from augmented data
    salaries_y_augmented = salaries_augmented["current_annual_salary"].skb.mark_as_y()
    salaries_X_augmented = salaries_augmented.skb.drop(["current_annual_salary"]).skb.mark_as_X()

    encoded = salaries_X_augmented.skb.apply(TableVectorizer())
    return encoded.skb.apply(GradientBoostingRegressor(random_state=0), y=salaries_y_augmented)


def _create_env(X, y, operator_name, state):
    """Create environment dictionary for learner."""
    return {
        "_skrub_X": X,
        "_skrub_y": y,
        f"sempipes_memory__{operator_name}": None,
        f"sempipes_pipeline_summary__{operator_name}": None,
        f"sempipes_prefitted_state__{operator_name}": state,
    }


def test_sem_augment_optimizable():
    ensure_default_config()

    # Fetch both train and test splits
    salaries_df_train = skrub.datasets.fetch_employee_salaries(split="train").employee_salaries
    X_train = salaries_df_train.drop(columns="current_annual_salary", errors="ignore")
    y_train = salaries_df_train["current_annual_salary"]

    salaries_df_test = skrub.datasets.fetch_employee_salaries(split="test").employee_salaries
    X_test = salaries_df_test.drop(columns="current_annual_salary", errors="ignore")
    y_test = salaries_df_test["current_annual_salary"]

    operator_name = "salary_augmentation"

    # Baseline w/o optimization
    baseline_pipeline = _pipeline(salaries_df_train, operator_name)
    data_op = find_node_by_name(baseline_pipeline, operator_name)
    empty_state = data_op._skrub_impl.estimator.empty_state()

    learner = baseline_pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(_create_env(X_train, y_train, operator_name, empty_state))
    before_optimization = learner.score(_create_env(X_test, y_test, operator_name, empty_state))

    # Optimize the augmentation operator on train data
    pipeline_to_optimise = _pipeline(salaries_df_train, operator_name)
    outcomes = optimise_colopro(
        pipeline_to_optimise,
        operator_name,
        num_trials=3,
        scoring="r2",
        search=EvolutionarySearch(population_size=2),
        cv=2,
        num_hpo_iterations_per_trial=2,
        pipeline_definition=_pipeline,
    )

    best_outcome = max(outcomes, key=lambda x: x.score)
    print(f"Best outcome score after optimization on train CV: {best_outcome.score}, state: {best_outcome.state}")

    learner_optimized = pipeline_to_optimise.skb.make_learner(fitted=False, keep_subsampling=False)
    learner_optimized.fit(_create_env(X_train, y_train, operator_name, best_outcome.state))
    after_optimization = learner_optimized.score(_create_env(X_test, y_test, operator_name, best_outcome.state))

    print(f"Before optimization: {before_optimization}, after optimization: {after_optimization}")
    assert before_optimization <= after_optimization
