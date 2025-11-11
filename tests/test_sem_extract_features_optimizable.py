import skrub

from sempipes.optimisers import EvolutionarySearch, optimise_colopro


def test_sem_extract_features_text():
    # Fetch a dataset
    dataset_df = skrub.datasets.fetch_toxicity()
    dataset = skrub.var("toxicity", dataset_df)

    X = dataset.X.skb.mark_as_X()
    y = dataset.y.skb.mark_as_y()

    pipeline = skrub.tabular_pipeline("classification")
    task = X.skb.apply(pipeline, y=y)
    split = task.skb.train_test_split(random_state=0)
    learner = task.skb.make_learner()
    learner.fit(split["train"])
    score_before = learner.score(split["test"])

    # Define the target columns
    X = X.sem_extract_features(
        nl_prompt="Extract up to five features from the text tweets that could help predict toxicity. Focus on sentiment, presence of hate speech, and any other relevant linguistic features. If you encounter neutral or not valid content like a link, treat as a no sentiment.",
        input_columns=["text"],
        generate_via_code=True,
        name="toxicity_text_features",
    )

    pipeline_with_features = skrub.tabular_pipeline("classification")
    task_with_features = X.skb.apply(pipeline_with_features, y=y)
    split_with_features = task_with_features.skb.train_test_split(random_state=0)
    learner_with_features = task_with_features.skb.make_learner()
    learner_with_features.fit(split_with_features["train"])
    score_with_features = learner_with_features.score(split_with_features["test"])

    outcomes = optimise_colopro(
        task_with_features,
        "toxicity_text_features",
        num_trials=3,
        scoring="accuracy",
        search=EvolutionarySearch(population_size=6),
        cv=3,
        num_hpo_iterations_per_trial=1,
    )

    best_outcome = max(outcomes, key=lambda x: x.score)
    print(f"Best outcome (train) score after optimization: {best_outcome.score}")

    pipeline = task_with_features
    learner = pipeline.skb.make_learner(fitted=True, keep_subsampling=False)
    score_with_features_optimized = learner.score(split_with_features["test"])

    print(f"Tabular predictor performance w/o extracted features: {score_before}")
    print(f"Tabular predictor performance with extracted features: {score_with_features}")
    print(f"Tabular predictor performance with extracted features (optimized): {score_with_features_optimized}")

    assert score_before <= score_with_features <= score_with_features_optimized
