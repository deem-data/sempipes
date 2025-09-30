import numpy as np
import pandas as pd
import skrub

import sempipes  # pylint: disable=unused-import


def test_sem_deduplicate():
    # Fetch a dataset
    sport_products = pd.read_csv("tests/data/sports.csv").head(n=100)
    column_to_deduplicate = "GeneratedProductType"

    print("Unique values before the deduplication: ", sport_products[column_to_deduplicate].unique())

    sport_products_ref = skrub.var("sport_products", sport_products)

    sport_products_ref = sport_products_ref.sem_deduplicate(
        nl_prompt="Given a column with product types, please reorganise this column by removing duplicates or grouping similar items. For example, 'soccer ball' and 'football' can be grouped into 'sports ball'.",
        target_column=column_to_deduplicate,
        deduplicate_with_existing_values_only=False,
    ).skb.eval()

    print("Unique values after the deduplication: ", sport_products_ref[column_to_deduplicate].unique())

    correspondences_new_to_old = pd.concat(
        [sport_products[column_to_deduplicate], sport_products_ref[column_to_deduplicate]],
        axis=1,
        keys=[column_to_deduplicate + "1", column_to_deduplicate + "2"],
    ).drop_duplicates()

    correspondences_new_to_all_old = (
        correspondences_new_to_old.groupby(column_to_deduplicate + "1")[column_to_deduplicate + "2"]
        .apply(list)
        .to_dict()
    )

    print("Before and after value mapping: ", correspondences_new_to_all_old)

    expected_deplications = {
        "ball": ["sports ball"],
        "basketball": ["sports ball"],
        "colorful polka dot umbrella": ["umbrella"],
        "soccer ball": ["sports ball"],
        "sports ball": ["sports ball"],
    }

    correct_deduplicated_values = True
    for deduplicated_item in correspondences_new_to_all_old:
        if (
            deduplicated_item in expected_deplications
            and expected_deplications[deduplicated_item] != correspondences_new_to_all_old[deduplicated_item]
        ):
            correct_deduplicated_values = False
            print(
                f"Deduplication for item {deduplicated_item} is not as expected. Expected {expected_deplications[deduplicated_item]}, got {correspondences_new_to_all_old[deduplicated_item]}."
            )

    assert correct_deduplicated_values, "The deduplicated values are not as expected."
    assert sport_products_ref[column_to_deduplicate].isna().sum() == 0


def test_deduplication_typos():
    # Create a dataset
    country_codes = [
        "US",
        "usa",
        "USA ",
        " US",
        "U-S",
        "uk",
        "UK ",
        "GB",
        "FR",
        "fra",
        "FRA ",
        "FRN",
        "DE",
        "deu",
        "GER",
        "SP",
        "es",
        "ES ",
        "ESp",
        "CN",
        "CHN",
        " Cn",
        "CA",
        "cA ",
        "CAN",
        "JP",
        "jp",
        " JPN",
        "USA",
        "DE",
        "ESP",
        "UK",
    ]

    countries = pd.DataFrame(country_codes, columns=["country_code"])
    column_to_deduplicate = "country_code"

    print("Unique values before the deduplication: ", countries[column_to_deduplicate].unique())

    countries_ref = skrub.var("country_codes", countries)

    countries_ref = countries_ref.sem_deduplicate(
        nl_prompt="Given a column with country codes like USA, FR. Please reorganise this column by removing duplicates or typos.",
        target_column=column_to_deduplicate,
        deduplicate_with_existing_values_only=True,
    ).skb.eval()

    print("Unique values after the deduplication: ", countries_ref[column_to_deduplicate].unique())

    correspondences_new_to_old = pd.concat(
        [countries[column_to_deduplicate], countries_ref[column_to_deduplicate]],
        axis=1,
        keys=[column_to_deduplicate + "1", column_to_deduplicate + "2"],
    ).drop_duplicates()

    correspondences_new_to_all_old = (
        correspondences_new_to_old.groupby(column_to_deduplicate + "1")[column_to_deduplicate + "2"]
        .apply(list)
        .to_dict()
    )
    print("Before and after value mapping: ", correspondences_new_to_all_old)

    correct_deduplicated_values = np.all(
        [
            deduplicated_code in {"US", "UK", "FR", "DE", "ES", "CN", "CA", "JP", "JPN"}
            for deduplicated_code in list(countries_ref[column_to_deduplicate].unique())
        ]
    )

    assert correct_deduplicated_values, "The deduplicated values are not as expected."
    assert countries_ref[column_to_deduplicate].isna().sum() == 0
    assert countries_ref[column_to_deduplicate].nunique() < countries[column_to_deduplicate].nunique()


def test_deduplication_cities():
    # Create a dataset
    cities = pd.DataFrame(
        [
            "Rome, Italy",
            "Rome",
            "Roma, Italia",
            "Madrid, SP",
            "Madrid, spain",
            "Madrid",
            "Romq",
            "Rome, It",
        ],
        columns=["city"],
    )
    column_to_deduplicate = "city"

    print("Unique values before the deduplication: ", cities[column_to_deduplicate].unique())

    cities_ref = skrub.var("cities", cities)

    cities_ref = cities_ref.sem_deduplicate(
        nl_prompt="Given a column with city names, please reorganise this column by removing duplicates or typos. Remove country names.",
        target_column=column_to_deduplicate,
        deduplicate_with_existing_values_only=False,
    ).skb.eval()

    print("Unique values after the deduplication: ", cities_ref[column_to_deduplicate].unique())

    correspondences_new_to_old = pd.concat(
        [cities[column_to_deduplicate], cities_ref[column_to_deduplicate]],
        axis=1,
        keys=[column_to_deduplicate + "1", column_to_deduplicate + "2"],
    ).drop_duplicates()

    correspondences_new_to_all_old = (
        correspondences_new_to_old.groupby(column_to_deduplicate + "1")[column_to_deduplicate + "2"]
        .apply(list)
        .to_dict()
    )

    correct_deduplicated_values = np.all(
        [
            deduplicated_city in {"Rome", "Madrid"}
            for deduplicated_city in list(cities_ref[column_to_deduplicate].unique())
        ]
    )

    print("Before and after value mapping: ", correspondences_new_to_all_old)

    assert correct_deduplicated_values, "The deduplicated values are not as expected. Expected ['Rome', 'Madrid']."
    assert cities_ref[column_to_deduplicate].isna().sum() == 0
    assert cities_ref[column_to_deduplicate].nunique() < cities[column_to_deduplicate].nunique()
