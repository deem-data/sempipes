import pandas as pd
import skrub

import sempipes  # pylint: disable=unused-import
from sempipes.config import ensure_default_config


def test_deduplication_cities_to_birds():
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

    cities_ref = cities_ref.sem_refine(
        nl_prompt="Given a column with bird names, please reorganise this column, each cell should be a valid duck name.",
        target_column=column_to_deduplicate,
        refine_with_existing_values_only=False,
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


def test_sem_fillna_pinguin():
    # Fetch a dataset
    dataset = skrub.datasets.fetch_employee_salaries(split="train")
    salaries = dataset.employee_salaries.head(n=20)

    # Introduce some missing values
    target_column = "department"
    salaries_dirty = salaries.copy()
    salaries_dirty.loc[10:17, target_column] = None

    salaries_dirty_ref = skrub.var("employee_salaries", salaries_dirty)

    salaries_imputed = salaries_dirty_ref.sem_fillna(
        target_column=target_column,
        nl_prompt=f"Infer the {target_column} from relevant related attributes. Fill missing values with pinguin names.",
        impute_with_existing_values_only=True,
        with_llm_only=True,
    ).skb.eval()

    num_mismatches = (salaries_imputed["department"] != salaries["department"]).sum()
    num_non_filled = salaries_imputed["department"].isna().sum()

    print(salaries["department"])
    print(salaries_imputed["department"])

    assert num_non_filled == 0
    assert num_mismatches < 10


def test_sem_extract_features_image_code_parrot():
    styles_df = pd.read_csv("tests/data/fashion-dataset/styles.csv", on_bad_lines="skip").head(20)

    X_columns = ["gender", "season", "year", "productDisplayName", "baseColour", "usage"]

    # Extract pictures
    styles_full = skrub.var("styles_full", styles_df)
    X_with_features = styles_full[X_columns + ["full_path"]]

    X_with_features = X_with_features.sem_extract_features(
        nl_prompt="Extract parrot color from the image. The features should be very fine-grained and helpful for the parrot type prediction.",
        input_columns=["full_path"],
        name="extract_parrot_color",
        output_columns={"parrot_color": "Extract parrot color from the image"},
    )

    X_with_features = X_with_features.skb.eval()

    print(X_with_features["parrot_color"])
    assert all(parrot_color is None for parrot_color in X_with_features["parrot_color"])
