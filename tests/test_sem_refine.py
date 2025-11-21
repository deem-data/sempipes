import pandas as pd
import skrub

import sempipes  # pylint: disable=unused-import
from sempipes.config import ensure_default_config


def test_sem_refine():
    ensure_default_config()

    sport_products = pd.read_csv("tests/data/sports.csv").head(n=100)
    column_to_refine = "ProductType"

    print("Unique values before the refinement: ", sport_products[column_to_refine].unique())

    sport_products_ref = skrub.var("sport_products", sport_products)

    sport_products_ref = sport_products_ref.sem_refine(
        nl_prompt="Given a column with product types, please reorganise this column by removing duplicates or grouping similar items. For example, 'soccer ball' and 'football' can be grouped into 'sports ball'.",
        target_column=column_to_refine,
        refine_with_existing_values_only=False,
    ).skb.eval()

    print("Unique values after the refinement: ", sport_products_ref[column_to_refine].unique())

    correspondences_new_to_old = pd.concat(
        [sport_products[column_to_refine], sport_products_ref[column_to_refine]],
        axis=1,
        keys=[column_to_refine + "1", column_to_refine + "2"],
    ).drop_duplicates()

    correspondences_new_to_all_old = (
        correspondences_new_to_old.groupby(column_to_refine + "1")[column_to_refine + "2"].apply(list).to_dict()
    )

    print("Before and after value mapping: ", correspondences_new_to_all_old)

    expected_deplications = {
        "ball": ["sports ball"],
        "basketball": ["sports ball"],
        "colorful polka dot umbrella": ["umbrella"],
        "soccer ball": ["sports ball"],
        "sports ball": ["sports ball"],
    }

    correct_refined_values = True
    for refined_item in correspondences_new_to_all_old:
        if (
            refined_item in expected_deplications
            and expected_deplications[refined_item] != correspondences_new_to_all_old[refined_item]
        ):
            correct_refined_values = False
            print(
                f"Refinement for item {refined_item} is not as expected. Expected {expected_deplications[refined_item]}, got {correspondences_new_to_all_old[refined_item]}."
            )

    assert correct_refined_values, "The refined values are not as expected."
    assert sport_products_ref[column_to_refine].isna().sum() == 0


def test_refinement_typos():
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
        "Hamburg",  # outlier and should be fixed to DE
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
        "ES",
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
        "DE",
    ]

    countries = pd.DataFrame(dirty_country_codes, columns=["country_code"])

    countries_ref = skrub.var("country_codes", countries)

    cleaned_countries_ref = countries_ref.sem_refine(
        nl_prompt="Make sure that all values are in the ISO 3166-1 alpha-2 two-letter uppercase format",
        target_column="country_code",
        refine_with_existing_values_only=False,
    ).skb.eval()

    cleaned_codes = list(cleaned_countries_ref["country_code"])
    code_pairs = list(zip(dirty_country_codes, cleaned_codes, correct_codes))

    mismatches = [
        (dirty, cleaned, correct)
        for dirty, cleaned, correct in code_pairs
        if cleaned != correct and not (cleaned == "UK" and correct == "GB")
    ]

    assert len(mismatches) < 3, f"Cleaning failed for too many country codes: {mismatches}"


def test_refinement_cities():
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
    column_to_refine = "city"

    print("Unique values before the refinement: ", cities[column_to_refine].unique())

    cities_ref = skrub.var("cities", cities)

    cities_ref = cities_ref.sem_refine(
        nl_prompt="Given a column with city names, please reorganise this column by removing duplicates or typos. Remove country names.",
        target_column=column_to_refine,
        refine_with_existing_values_only=False,
    ).skb.eval()

    print("Unique values after the refinement: ", cities_ref[column_to_refine].unique())

    correspondences_new_to_old = pd.concat(
        [cities[column_to_refine], cities_ref[column_to_refine]],
        axis=1,
        keys=[column_to_refine + "1", column_to_refine + "2"],
    ).drop_duplicates()

    correspondences_new_to_all_old = (
        correspondences_new_to_old.groupby(column_to_refine + "1")[column_to_refine + "2"].apply(list).to_dict()
    )

    refined_cities = set(cities_ref[column_to_refine].unique())

    print("Before and after value mapping: ", correspondences_new_to_all_old)

    assert {
        "Rome",
        "Madrid",
    } == refined_cities, f"Expected ['Rome', 'Madrid'] after refinement, but got {refined_cities}."
    assert cities_ref[column_to_refine].isna().sum() == 0
    assert cities_ref[column_to_refine].nunique() < cities[column_to_refine].nunique()
