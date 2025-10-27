import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

import sempipes

sempipes.update_config(
    llm_for_batch_processing=sempipes.LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 0.0},
    ),
)

raw_countries = pd.read_csv("experiments/aide_failures_dataintegration/a.csv")
raw_country_stats = pd.read_csv("experiments/aide_failures_dataintegration/b.csv")

countries = skrub.var("countries", raw_countries)
country_stats = skrub.var("country_stats", raw_country_stats)

country_stats_with_codes = country_stats.sem_extract_features(
    nl_prompt="Generate the ISO 2 letter country code from the country name",
    input_columns=["nn"],
    output_columns={"ic": "the two letter country code"},
)

country_stats_and_gdp = countries.merge(country_stats_with_codes, on="ic")

gdp = country_stats_and_gdp["gdp"].skb.apply_func(np.log1p)
stats = country_stats_and_gdp.drop(columns=["gdp"])

stats = stats.skb.mark_as_X().skb.set_description("Economic, social and political country statistics")
gdp = gdp.skb.mark_as_y().skb.set_description("A country's GDP")

encoded_stats = stats.skb.apply(skrub.TableVectorizer(), exclude_cols=["ic", "nn"])
encoded_stats = encoded_stats.drop(columns=["ic", "nn"])
pipeline = encoded_stats.skb.apply(HistGradientBoostingRegressor(), y=gdp)

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])
    score = root_mean_squared_error(split["test"]["_skrub_y"], y_pred)
    print(f"RMSE on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
