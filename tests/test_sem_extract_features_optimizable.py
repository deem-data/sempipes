import skrub

from sempipes.optimisers import MonteCarloTreeSearch, optimise_colopro


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
    print(f"Tabular predictor performance w/o extracted features: {score_before}")

    # Define the target columns
    X = X.sem_extract_features(
        nl_prompt="Extract up to five features from the text tweets that could help predict toxicity. Focus on sentiment, presence of hate speech, and any other relevant linguistic features. If you encounter neutral or not valid content like a link, treat as a no sentiment.",
        input_columns=["text"],
        generate_via_code=True,
    )

    pipeline_with_features = skrub.tabular_pipeline("classification")
    task_with_features = X.skb.apply(pipeline_with_features, y=y)
    split_with_features = task_with_features.skb.train_test_split(random_state=0)
    learner_with_features = task.skb.make_learner()
    learner_with_features.fit(split_with_features["train"])
    score_with_features = learner_with_features.score(split_with_features["test"])
    print(f"Tabular predictor performance with extracted features: {score_with_features}")

    # Note that this should be done on a separate validation set in a real world use case
    outcomes = optimise_colopro(
        pipeline_with_features,
        "sem_extract_features",
        num_trials=5,
        scoring="accuracy",
        search=MonteCarloTreeSearch(nodes_per_expansion=2, c=1.41),
        cv=3,
        num_hpo_iterations_per_trial=1,  # 0,
    )

    best_outcome = max(outcomes, key=lambda x: x.score)
    assert score_before <= best_outcome.score


test_sem_extract_features_text()
