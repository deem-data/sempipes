import pandas as pd
import skrub

import sempipes  # pylint: disable=unused-import
from sempipes.config import ensure_default_config


def test_sem_deduplicate():
    ensure_default_config()

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
    ensure_default_config()

    dirty_country_codes = [
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

    correct_codes = [
        "US",
        "US",
        "US",
        "US",
        "US",
        "GB",
        "GB",
        "GB",
        "FR",
        "FR",
        "FR",
        "FR",
        "DE",
        "DE",
        "DE",
        "ES",
        "es",
        "ES",
        "ES",
        "CN",
        "CN",
        "CN",
        "CA",
        "CA",
        "CA",
        "JP",
        "JP",
        "JP",
        "US",
        "DE",
        "ES",
        "GB",
    ]

    countries = pd.DataFrame(dirty_country_codes, columns=["country_code"])

    countries_ref = skrub.var("country_codes", countries)

    cleaned_countries_ref = countries_ref.sem_deduplicate(
        nl_prompt="Make sure that all values are in the ISO 3166-1 alpha-2 two-letter uppercase format",
        target_column="country_code",
        deduplicate_with_existing_values_only=False,
    ).skb.eval()

    cleaned_codes = list(cleaned_countries_ref["country_code"])
    code_pairs = list(zip(dirty_country_codes, cleaned_codes, correct_codes))

    mismatches = [
        (dirty, cleaned, correct)
        for dirty, cleaned, correct in code_pairs
        if cleaned != correct and not (cleaned == "UK" and correct == "GB")
    ]

    assert len(mismatches) < 3, f"Cleaning failed for too many country codes: {mismatches}"


def test_deduplication_cities():
    ensure_default_config()

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

    deduplicated_cities = set(cities_ref[column_to_deduplicate].unique())

    print("Before and after value mapping: ", correspondences_new_to_all_old)

    assert {
        "Rome",
        "Madrid",
    } == deduplicated_cities, f"Expected ['Rome', 'Madrid'] after deduplication., but got {deduplicated_cities}."
    assert cities_ref[column_to_deduplicate].isna().sum() == 0
    assert cities_ref[column_to_deduplicate].nunique() < cities[column_to_deduplicate].nunique()
