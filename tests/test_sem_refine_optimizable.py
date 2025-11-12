import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from skrub import TableVectorizer
from skrub._data_ops._evaluation import find_node_by_name

import sempipes
from sempipes.config import ensure_default_config
from sempipes.optimisers import EvolutionarySearch, optimise_colopro


def _pipeline(products_df, baskets_df, operator_name):
    products = skrub.var("products", products_df)
    baskets = skrub.var("baskets", baskets_df)

    basket_ids = sempipes.as_X(baskets[["ID"]], "Shopping baskets with product transactions")
    fraud_flags = sempipes.as_y(baskets["fraud_flag"], "A binary flag indicating a fraudulent shopping basket")

    products_refined = products.sem_refine(
        nl_prompt="Normalize manufacturer names by fixing case variations (e.g., 'APPLE' vs 'Apple' vs 'apple'), handling abbreviations (e.g., 'HP' vs 'Hewlett-Packard'), and unifying similar brand names, maybe rework brand names into something more general like 'electronics', 'household', 'toys', 'clothes', 'books', 'music', 'movies', 'games', 'software', 'hardware', 'other'. Standardize naming conventions. No complex or repetitive code.",
        target_column="make",
        refine_with_existing_values_only=False,
        name=operator_name,
    )

    vectorizer = TableVectorizer()
    vectorized_products = products_refined.skb.apply(vectorizer, exclude_cols="basket_ID")

    aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
    augmented_baskets = basket_ids.merge(aggregated_products, left_on="ID", right_on="basket_ID").drop(
        columns=["ID", "basket_ID"]
    )

    return augmented_baskets.skb.apply(HistGradientBoostingClassifier(random_state=0), y=fraud_flags)


def _pipeline_no_sempipes(products_df, baskets_df):
    products = skrub.var("products", products_df)
    baskets = skrub.var("baskets", baskets_df)

    basket_ids = baskets[["ID"]]
    fraud_flags = baskets["fraud_flag"]

    vectorizer = skrub.TableVectorizer()
    vectorized_products = products.skb.apply(vectorizer, exclude_cols="basket_ID")

    aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
    augmented_baskets = basket_ids.merge(aggregated_products, left_on="ID", right_on="basket_ID").drop(
        columns=["ID", "basket_ID"]
    )

    return augmented_baskets.skb.apply(HistGradientBoostingClassifier(random_state=0), y=fraud_flags)


def _create_env(products_df, baskets_df, operator_name, state):
    """Create environment dictionary for learner."""
    return {
        "products": products_df,
        "baskets": baskets_df,
        f"sempipes_memory__{operator_name}": None,
        f"sempipes_pipeline_summary__{operator_name}": None,
        f"sempipes_prefitted_state__{operator_name}": state,
    }


def _introduce_make_variations(df):
    """Introduce errors, misspellings, and variations in the make column to test refinement. This simulates real-world data inconsistencies."""
    make_mapping = {
        "APPLE": [
            "APPLE",
            "Apple",
            "apple",
            "APLE",
            "APLLE",
            "APPEL",
            "APLLE",
            "APPL",
            "APEL",
            "APPLE ",
            " Apple",
            "APPLe",
            "APLLE",
            "APEL",
        ],
        "SONY": ["SONY", "Sony", "sony", "SONNY", "SONI", "SONNY", "SONEY", "SON ", "Sony ", "SONI", "SONEY"],
        "MICROSOFT": [
            "MICROSOFT",
            "Microsoft",
            "microsoft",
            "MSFT",
            "MS",
            "MICROSOFT ",
            "MICROSOFT",
            "MICROSOF",
            "MICROSOFTT",
            "MICROSOFT ",
            " Microsoft",
            "MICROSOFTT",
        ],
        "HP": [
            "HP",
            "Hewlett-Packard",
            "HEWLETT PACKARD",
            "H.P.",
            "hp",
            "Hp",
            "H P",
            "HP ",
            " Hewlett Packard",
            "HEWLETT-PACKARD",
            "hewlett packard",
            "H.P",
        ],
        "LG": ["LG", "L.G.", "lg", "L G", "LG ", " L.G.", "Lg", "L-G", "L.G"],
        "SAMSUNG": [
            "SAMSUNG",
            "Samsung",
            "samsung",
            "SAMSUNG ",
            "SAMSUMG",
            "SAMSUGN",
            "SAMSUN",
            "SAMSUNG ",
            " Samsung",
            "SAMSUMG",
            "SAMSUGN",
        ],
        "PHILIPS": [
            "PHILIPS",
            "Philips",
            "philips",
            "PHILLIPS",
            "PHILIP",
            "PHILIPS ",
            "PHILIPPS",
            "PHILIP ",
            " Philips",
            "PHILLIPS",
        ],
        "PANASONIC": [
            "PANASONIC",
            "Panasonic",
            "panasonic",
            "PANASONIC ",
            "PANASOINC",
            "PANASONIC",
            "PANASONIK",
            "PANASOINC",
            " Panasonic",
            "PANASONIK",
        ],
        "DELL": ["DELL", "Dell", "dell", "DELL ", "DEL", "DELLL", "DEL ", " Dell", "DELL", "DELLL"],
        "LENOVO": [
            "LENOVO",
            "Lenovo",
            "lenovo",
            "LENOVO ",
            "LENOV",
            "LENOVO",
            "LENOVVO",
            "LENOV ",
            " Lenovo",
            "LENOVVO",
        ],
        "ASUS": ["ASUS", "Asus", "asus", "ASUS ", "ASU", "ASUSS", "ASUS ", " Asus", "ASU", "ASUSS"],
        "TOSHIBA": [
            "TOSHIBA",
            "Toshiba",
            "toshiba",
            "TOSHIBA ",
            "TOSHIB",
            "TOSHIBA",
            "TOSHIBBA",
            "TOSHIB ",
            " Toshiba",
            "TOSHIBBA",
        ],
        "CANON": ["CANON", "Canon", "canon", "CANON ", "CANNON", "CANON", "CANNO", "CANON ", " Canon", "CANNON"],
        "NIKON": ["NIKON", "Nikon", "nikon", "NKON ", "NIKKON", "NIKON", "NIKKON", "NIKON ", " Nikon", "NKON"],
        "BOSE": ["BOSE", "Bose", "bose", "BOSE ", "BOS", "BOSSE", "BOSE ", " Bose", "BOS", "BOSSE"],
        "SONOS": ["SONOS", "Sonos", "sonos", "SONOS ", "SONOSS", "SONOS", "SONOSS", "SONOS ", " Sonos", "SONOSS"],
        "GOOGLE": [
            "GOOGLE",
            "Google",
            "google",
            "GOOGLE ",
            "GOOGL",
            "GOOGLE",
            "GOOGLEE",
            "GOOGL ",
            " Google",
            "GOOGLEE",
        ],
        "AMAZON": [
            "AMAZON",
            "Amazon",
            "amazon",
            "AMAZON ",
            "AMAZN",
            "AMAZON",
            "AMAZONN",
            "AMAZN ",
            " Amazon",
            "AMAZONN",
        ],
        "NINTENDO": [
            "NINTENDO",
            "Nintendo",
            "nintendo",
            "NINTENDO ",
            "NINTENDOO",
            "NINTENDO",
            "NINTENDOO",
            "NINTENDO ",
            " Nintendo",
            "NINTENDOO",
        ],
        "LOGITECH": [
            "LOGITECH",
            "Logitech",
            "logitech",
            "LOGITECH ",
            "LOGITEC",
            "LOGITECH",
            "LOGITECHH",
            "LOGITEC ",
            " Logitech",
            "LOGITECHH",
        ],
        "BEATS": ["BEATS", "Beats", "beats", "BEATS ", "BEAT", "BEATTS", "BEATS ", " Beats", "BEATTS"],
        "BRAUN": ["BRAUN", "Braun", "braun", "BRAUN ", "BRAUNN", "BRAUN ", " Braun", "BRAUNN"],
        "SAGE": ["SAGE", "Sage", "sage", "SAGE ", "SAG", "SAGEE", "SAGE ", " Sage", "SAGEE"],
        "KITCHENAID": [
            "KITCHENAID",
            "KitchenAid",
            "kitchenaid",
            "KITCHENAID ",
            "KITCHEN AID",
            "KITCHENAID",
            "KITCHEN-AID",
            " KitchenAid",
            "KITCHEN AID",
        ],
        "SMEG": ["SMEG", "Smeg", "smeg", "SMEG ", "SMEGG", "SMEG ", " Smeg", "SMEGG"],
        "NESPRESSO": [
            "NESPRESSO",
            "Nespresso",
            "nespresso",
            "NESPRESSO ",
            "NESPRESO",
            "NESPRESSO",
            "NESPRESSO ",
            " Nespresso",
            "NESPRESO",
        ],
        "BOSCH": ["BOSCH", "Bosch", "bosch", "BOSCH ", "BOSCHH", "BOSCH ", " Bosch", "BOSCHH"],
        "SIEMENS": ["SIEMENS", "Siemens", "siemens", "SIEMENS ", "SIEMEN", "SIEMENS", "SIEMENS ", " Siemens", "SIEMEN"],
        "AEG": ["AEG", "Aeg", "aeg", "AEG ", "A.E.G.", "AEG ", " Aeg", "A.E.G"],
        "MIELE": ["MIELE", "Miele", "miele", "MIELE ", "MIELEE", "MIELE ", " Miele", "MIELEE"],
        "DYSON": ["DYSON", "Dyson", "dyson", "DYSON ", "DYSONN", "DYSON ", " Dyson", "DYSONN"],
        "SHARK": ["SHARK", "Shark", "shark", "SHARK ", "SHARKK", "SHARK ", " Shark", "SHARKK"],
        "VAX": ["VAX", "Vax", "vax", "VAX ", "VAXX", "VAX ", " Vax", "VAXX"],
    }
    df = df.copy()
    np.random.seed(42)

    for original, variations in make_mapping.items():
        # Find rows with the original make value
        mask = df["make"] == original
        if mask.sum() > 0:
            # Only introduce errors to a random portion (e.g., 40-60% of rows)
            error_rate = np.random.uniform(0.4, 0.6)
            n_rows_to_modify = max(1, int(mask.sum() * error_rate))

            # Randomly select which rows to modify
            matching_indices = df[mask].index.tolist()
            np.random.shuffle(matching_indices)
            indices_to_modify = matching_indices[:n_rows_to_modify]

            # Assign variations to the selected rows
            for i, idx in enumerate(indices_to_modify):
                if i < len(variations):
                    df.loc[idx, "make"] = variations[i]

    # Also introduce some general errors: extra whitespace, case variations
    # Only to a random portion of rows
    mask_random = np.random.random(len(df)) < 0.1  # 10% of rows get extra whitespace
    df.loc[mask_random, "make"] = df.loc[mask_random, "make"].astype(str).str.strip() + " "

    # Introduce some case variations to a random portion
    mask_case = np.random.random(len(df)) < 0.15  # 15% of rows get case variations
    df.loc[mask_case, "make"] = df.loc[mask_case, "make"].astype(str).str.title()

    return df


def _get_baseline_score(train_baskets, kept_products_train, test_baskets, kept_products_test, operator_name):
    """Get baseline score without optimization."""
    baseline_pipeline = _pipeline(kept_products_train, train_baskets, operator_name)
    data_op = find_node_by_name(baseline_pipeline, operator_name)
    empty_state = data_op._skrub_impl.estimator.empty_state()

    learner = baseline_pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(_create_env(kept_products_train, train_baskets, operator_name, empty_state))
    return learner.score(_create_env(kept_products_test, test_baskets, operator_name, empty_state))


def _get_baseline_score_no_sempipes(
    train_baskets, kept_products_train, test_baskets, kept_products_test, operator_name
):
    """Get baseline score without sempipes."""
    baseline_pipeline = _pipeline_no_sempipes(kept_products_train, train_baskets)

    learner = baseline_pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(_create_env(kept_products_train, train_baskets, operator_name, None))
    return learner.score(_create_env(kept_products_test, test_baskets, operator_name, None))


def _get_optimized_score(train_baskets, kept_products_train, test_baskets, kept_products_test, operator_name):
    """Get optimized score after optimization."""
    pipeline_to_optimise = _pipeline(kept_products_train, train_baskets, operator_name)
    outcomes = optimise_colopro(
        pipeline_to_optimise,
        operator_name,
        num_trials=3,
        scoring="f1",
        search=EvolutionarySearch(population_size=2),
        cv=2,
        num_hpo_iterations_per_trial=1,
        pipeline_definition=_pipeline,
    )

    best_outcome = max(outcomes, key=lambda x: x.score)
    print(f"Best outcome score after optimization on train CV: {best_outcome.score}, state: {best_outcome.state}")

    learner_optimized = pipeline_to_optimise.skb.make_learner(fitted=False, keep_subsampling=False)
    learner_optimized.fit(_create_env(kept_products_train, train_baskets, operator_name, best_outcome.state))
    return learner_optimized.score(_create_env(kept_products_test, test_baskets, operator_name, best_outcome.state))


def test_sem_refine_optimizable():
    ensure_default_config()

    # Fetch credit fraud dataset
    dataset = skrub.datasets.fetch_credit_fraud()
    products = dataset["products"]
    baskets = dataset["baskets"]
    baskets = baskets.sample(n=10000, replace=False, random_state=42)

    products_with_errors = _introduce_make_variations(products)

    train_baskets, test_baskets = train_test_split(baskets, test_size=0.3, random_state=42)
    kept_products_train = products_with_errors[products_with_errors["basket_ID"].isin(train_baskets["ID"])]
    kept_products_test = products_with_errors[products_with_errors["basket_ID"].isin(test_baskets["ID"])]

    operator_name = "make_refinement"

    # Baseline w/o sempipes
    clean_products_train = products[products["basket_ID"].isin(train_baskets["ID"])]
    clean_products_test = products[products["basket_ID"].isin(test_baskets["ID"])]
    baseline_score_no_sempipes = _get_baseline_score_no_sempipes(
        train_baskets, clean_products_train, test_baskets, clean_products_test, operator_name
    )

    # Baseline w/o optimization
    before_optimization = _get_baseline_score(
        train_baskets, kept_products_train, test_baskets, kept_products_test, operator_name
    )

    # Optimized w/ optimization
    after_optimization = _get_optimized_score(
        train_baskets, kept_products_train, test_baskets, kept_products_test, operator_name
    )

    print(
        f"Baseline w/o SemPipes: {baseline_score_no_sempipes}.With SemPipes.sem_refine: before optimization: {before_optimization}, after optimization: {after_optimization}"
    )
    assert before_optimization <= after_optimization


test_sem_refine_optimizable()
