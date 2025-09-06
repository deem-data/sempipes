# Gyyre - Context-Aware Semantic Operators for Machine Learning Pipelines

Gyyre is a research project to extend Python-based machine learning scripts with semantic operators. It is heavily relying on the awesome work from the [skrub](https://github.com/skrub-data/skrub) project!

### Semantic Operators

 - `sem_choose(nl_prompt)` -- a semantic drop-in alternative for [skrub's choose_from](https://skrub-data.org/stable/reference/generated/skrub.choose_from.html) to suggest hyperparameter ranges and other pipeline components
 - `sem_fillna(target_column, nl_prompt: str, impute_with_existing_values_only)` -- missing value imputation
 - `with_sem_features(nl_prompt, how_many)` -- automated generation of additional feature columns in dataframes
 - `sem_select(nl_prompt)`  -- a semantic drop-in alternative for [skrub's selectors](https://skrub-data.org/stable/userguide_selectors.html) to select columns from dataframes
   
### Example

```python
import gyyre
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from gyyre import sem_choose

dataset = skrub.datasets.fetch_credit_fraud()

products = skrub.var("products", dataset.products)
baskets = skrub.var("baskets", dataset.baskets)
baskets = baskets.skb.subsample(n=5000, how="random")

basket_ids = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

# Impute missing values in your data
products = products.sem_fillna(
    target_column="make",
    nl_prompt="Infer the manufacturer from relevant product-related attributes like title or description.",
    impute_with_existing_values_only=True,
)

kept_products = products[products["basket_ID"].isin(basket_ids["ID"])]
# Generate new features for the model to train
kept_products = kept_products.with_sem_features(
    nl_prompt="""
    Generate additional brand- and manufacturer-related product features. Make sure that they can be
    efficiently computed on large datasets, and that they work across a large number of brands and
    manufacturers. Use your intrinsic knowledge about what products and brands fraudsters focus on
    to make sure that the new features are helpful for the prediction task  at hand.
    """,
    name="brand_features",
    how_many=5,
)

vectorizer = skrub.TableVectorizer()
vectorized_products = kept_products.skb.apply_with_sem_choose(
    vectorizer,
    exclude_cols="basket_ID",
    # Choose encoders for your data
    choices=sem_choose(
        high_cardinality="""
        A fast encoder for messy columns with potentially invalid data that can scale to many unique
        values, can handle missing values and that outputs a pandas Dataframe as result.
    """
    ),
)

aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
augmented_baskets = basket_ids.merge(aggregated_products, left_on="ID", right_on="basket_ID").drop(
    columns=["ID", "basket_ID"]
)

hgb = HistGradientBoostingClassifier()
fraud_detector = augmented_baskets.skb.apply_with_sem_choose(
    hgb,
    y=fraud_flags,
    # Get suggestions for hyperparameters
    choices=sem_choose(learning_rate="A range of reasonable learning rates to try")
)
```
