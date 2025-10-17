import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error

import sempipes

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="openai/gpt-5",
        # parameters={ "temperature": 0.0},
    )
)


def sempipes_pipeline(data_file: str, pipeline_seed):
    np.random.seed(pipeline_seed)
    data = pd.read_csv(data_file)
    revenue_df = data.revenue
    data = data.drop(columns=["revenue"])

    movie_stats = skrub.var("movie_stats", data).skb.mark_as_X().skb.subsample(n=100)
    movie_stats = movie_stats.skb.set_description("""
    In a world… where movies made an estimated $41.7 billion in 2018, the film industry is more popular than ever. But what movies make the most money at the box office? How much does a director matter? Or the budget? For some movies, it's "You had me at 'Hello.'" For others, the trailer falls short of expectations and you think "What we have here is a failure to communicate." In this competition, you're presented with metadata on several thousand past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. You can collect other publicly available data to use in your model predictions, but in the spirit of this competition, use only data that would have been available before a movie's release. It is your job to predict the international box office revenue for each movie. For each id in the test set, you must predict the value of the revenue variable. Submissions are evaluated on Root-Mean-Squared-Logarithmic-Error (RMSLE) between the predicted value and the actual revenue. Logs are taken to not overweight blockbuster revenue movies.
    """)

    revenue = skrub.var("revenue", revenue_df).skb.mark_as_y().skb.subsample(n=100)
    revenue = revenue.skb.set_description("the international box office revenue for a movie")

    y_log = revenue.skb.apply_func(np.log1p)

    movie_stats = movie_stats.with_sem_features(
        nl_prompt="""
        Create additional features that could help predict the box office revenue of a movie. Here are detailed instructions:
        
        Compute the year, month, day, day of week, day of year of the movie release data, and the elapsed time since then
    
        Create a categorical feature for all of the following:        
        - each production company that has produced more than 30 movies
        - each production country that has hosted more than 10 movies
        - each languages that is spoken in more than 10 movies
        - each keyword that appears in more than 30 movies
        
        Compute the following features:
        - number of cast members
        - number of crew members
        - whether the movie has a tagline
        - length of the tagline (in words)
        - whether the movie has an overview
        - length of the overview (in words)
        
        Finally, extract the following metadata:
        - director of the movie
        - screenplay writer
        - director of photography
        - original music composer
        - art director       
        """,
        name="additional_movie_features",
    )

    to_remove = ["imdb_id", "id", "poster_path", "overview", "homepage", "tagline", "original_title", "status", "crew"]
    movie_stats = movie_stats.drop(to_remove, axis=1)

    encoder = skrub.TableVectorizer()
    X = movie_stats.skb.apply(encoder)

    feature_selector = SelectFromModel(
        RandomForestRegressor(
            n_estimators=40, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=True, random_state=42
        ),
        threshold=0.002,
        importance_getter="feature_importances_",
    )

    X = X.skb.apply(feature_selector, y=y_log)

    regressor = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=10, max_features=0.5, random_state=pipeline_seed, n_jobs=-1
    )

    predictions = X.skb.apply(regressor, y=y_log)
    return predictions


scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    pipeline = sempipes_pipeline("comparisons/tmdb_box_office_prediction/data.csv", seed)

    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    train_env = split["train"]
    test_env = split["test"]

    learner.fit(train_env)
    y_pred = learner.predict(test_env)

    rmsle = np.sqrt(mean_squared_error(np.log1p(test_env["_skrub_y"]), y_pred))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
