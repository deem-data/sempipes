import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import skrub
from tqdm import tqdm

import sempipes
from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label


def extract_x2_features_sempipes(X2_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from X2 dataset using sempipes and sem_extract_features.
    Returns a DataFrame with extracted features merged with original data.
    """
    data_ref = skrub.var("data_original_x2", X2_data).skb.mark_as_X()
    data_ref = data_ref.skb.set_description(
        "This is a dataset of product titles of removable storage devices "
        "(USB sticks, SD / microSD cards, SSDs, memory cards, and sometimes Samsung phones / TVs)."
    )

    brands_list = [
        "intenso",
        "lexar",
        "logilink",
        "pny",
        "samsung",
        "sandisk",
        "kingston",
        "sony",
        "toshiba",
        "transcend",
    ]
    families_dict = {
        "sandisk": [
            "cruizer",
            "tarjeta",
            "glide",
            "select",
            "extern",
            "origin",
            "transmemory",
            "react",
            "memo",
            "kart",
            "pendrive",
            "car",
            "serie",
            "line",
            "extreme",
            "cruzer",
            "ultra",
            "micro",
            "traveler",
            "hyperx",
            "adapt",
            "wex",
            "flash",
        ],
        "lexar": [
            "ultra",
            "xqd",
            "jumpdrive",
            "micro",
            "pendrive",
            "sd",
            "tarjeta",
            "memo",
            "usb",
            "extreme",
            "blade",
            "car",
            "scheda",
            "veloc",
            "react",
            "adapt",
            "secure",
            "premium",
            "wex",
            "transmemo",
            "alu",
            "datatravel",
            "canvas",
            "flair",
            "hyperx",
            "cruzer",
            "flash",
        ],
        "toshiba": [
            "ultra",
            "exceria",
            "traveler",
            "sdhc",
            "memoria",
            "xqd",
            "line",
            "usb",
            "transmemo",
            "extreme",
            "flair",
            "micro",
            "speicher",
            "serie",
            "car",
        ],
        "kingston": [
            "traveler",
            "cart",
            "adapt",
            "extreme",
            "memo",
            "canvas",
            "datatravel",
            "hyperx",
            "kart",
            "blade",
            "ultimate",
        ],
        "sony": [
            "extreme",
            "usm32gqx",
            "micro",
            "sd",
            "usb",
            "ultra",
            "jumpdrive",
            "hyperx",
            "memo",
            "kart",
            "xqd",
            "pendrive",
            "adapt",
            "blade",
            "cruzer",
            "flair",
            "glide",
            "cart",
            "tarjeta",
            "flash",
        ],
        "intenso": [
            "cs/ultra",
            "premium",
            "ultra",
            "micro",
            "line",
            "scheda",
            "usb",
            "sd",
            "tarjeta",
            "kart",
            "car",
            "transmemo",
        ],
        "pny": ["attach", "usb", "sd", "micro", "premium", "memo"],
        "samsung": [
            "galaxy",
            "speicher",
            "micro",
            "usb",
            "sd",
            "evo",
            "ultra",
            "extreme",
            "memo",
            "adapt",
            "car",
            "kart",
            "klasse",
            "multi",
            "jumpdrive",
            "flash",
        ],
        "transcend": [],
    }

    intenso_type = [
        "basic",
        "rainbow",
        "high speed",
        "speed",
        "premium",
        "alu",
        "business",
        "micro",
        "imobile",
        "cmobile",
        "mini",
        "ultra",
        "slim",
        "flash",
        "mobile",
    ]

    colors = [
        "midnight black",
        "prism white",
        "prism black",
        "prism green",
        "prism blue",
        "canary yellow",
        "flamingo pink",
        "cardinal red",
        "smoke blue",
        "deep blue",
        "coral orange",
        "black sky",
        "gold sand",
        "blue mist and peach cloud",
        "orchid gray",
        "metallic copper",
        "lavender purple",
        "ocean blue",
        "pure white",
        "alpine white",
        "copper",
        "red",
        "black",
        "blue",
        "white",
        "silver",
        "gold",
        "violet",
        "purple",
        "brown",
        "orange",
        "coral",
        "pink",
    ]

    output_columns = {
        "normalized_name": (
            "Canonical lowercase product title for entity blocking. "
            "Transliterate accents to ASCII, strip HTML/noise, normalize whitespace. "
            "Key normalizations: classe/clase/klasse -> class; uhs-i/uhs1 -> uhsi; "
            "type-c/usb-c -> type-c; hyperx/savage -> kingston hxs."
        ),
        "brand": (
            f"Storage device manufacturer. Known brands: {brands_list}. "
            "Handle typos: san disk/sandisc -> sandisk, samsun -> samsung. "
            "Lowercase. '0' if none."
        ),
        "capacity": (
            "Storage capacity in canonical form (32gb, 1tb). "
            "Normalize multilingual units: go/gigaoctet -> gb, to -> tb. "
            "'0' if none."
        ),
        "mem_type": (
            "Product category from title vocabulary: lte -> phone; xqd, ssd, tv, fdrive, memcard. "
            "Single token. '0' if none."
        ),
        "type": (
            f"Product line or family. Brand families: {families_dict}. "
            f"Intenso lines: {intenso_type}. Samsung colors: {colors}. "
            "cruizer -> cruzer. Examples: extreme, datatraveler, premium. '0' if none."
        ),
        "model": (
            "Model identifier: alphanumeric or hyphenated code, >=3 chars. "
            "Examples: dt101g2, u202, evo plus. '0' if none."
        ),
        "model_long": (
            "Manufacturer SKU or part number. "
            "Examples: sdcz50-064g-b35, usm32gqx, MB-MG32DA/EU. '0' if none."
        ),
        "model_short": (
            "Brief model codes like sda10, dt101, g1ux, u202. '0' if none."
        ),
        "features": (
            "Technical specs in canonical form: usb2/usb3, type-c, uhsi, class10, sdhc/sdxc, 95r80w. "
            "Normalize multilingual variants. '0' if none."
        ),
        "item_code": (
            "Numeric catalog code, often in parentheses: (4187407) -> 4187407. "
            "Prefer longer codes. '0' if none."
        ),
        "series": (
            f"Product series token. Brand families: {families_dict}. "
            f"Intenso: {intenso_type}. Samsung colors: {colors}. "
            "Examples: cruzer, glide, jumpdrive. Single token. '0' if none."
        ),
        "pat_hb": (
            "First hyphenated pattern: uhs-i, type-c, class-10, micro-sd. '0' if none."
        ),
        "hybrid": (
            "Letter-digit product codes >=5 chars: dt101g2, usm32gqx, sda10. '0' if none."
        ),
        "long_num": (
            "4+ digit sequences: 4187407, 483394661. '0' if none."
        ),
    }

    data_ref = data_ref.sem_extract_features(
        nl_prompt=(
            "YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. "
            "Use ONLY regex and rule-based approaches. "
            "DO NOT USE ANY Transformer, NER, LLM fallbacks, or LM models.\n\n"
            "Multilingual European e-commerce titles for storage devices (USB, SD, SSD). "
            "Extract stable blocking features across language and encoding differences.\n\n"
            "Prefer extracting over missing. Use '0' for absent values. All lowercase."
        ),
        input_columns=["name"],
        name="extract_x2_features",
        output_columns=output_columns,
        generate_via_code=True,
    )

    def fix_up(df):
        required_cols = [
            "id",
            "brand",
            "capacity",
            "normalized_name",
            "mem_type",
            "type",
            "model",
            "model_long",
            "model_short",
            "features",
            "item_code",
            "series",
            "pat_hb",
            "hybrid",
            "long_num",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "0"

        for col in required_cols:
            if col != "id":
                df[col] = df[col].fillna("0").astype(str).str.lower()

        return df

    data_ref = data_ref.skb.apply_func(fix_up)

    print("Discovering additional helpful features using sem_gen_features...")
    try:
        data_ref = data_ref.sem_gen_features(
            nl_prompt=(
                "YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. "
                "Use ONLY regex and rule-based approaches. "
                "DO NOT USE ANY Transformer, NER, LLM fallbacks, or LM models.\n\n"
                "Multilingual storage device titles for entity blocking.\n\n"
                "Already extracted: brand, capacity, normalized_name, mem_type, type, model, "
                "model_long, model_short, features, item_code, series, pat_hb, hybrid, long_num.\n\n"
                "Suggest additional blocking features. Lowercase strings, '0' for missing."
            ),
            name="discover_additional_blocking_features",
            how_many=5,
        )
        print("Additional features discovered successfully.")
    except Exception as e:
        print(f"Warning: sem_gen_features failed, continuing without it: {e}")

    USE_DISCOVERED_FEATURES = False

    if USE_DISCOVERED_FEATURES:
        print("Discovering additional helpful features for blocking...")

        existing_cols = [
            "id",
            "name",
            "normalized_name",
            "brand",
            "capacity",
            "type",
            "model",
            "model_long",
            "model_short",
            "features",
        ]
        existing_cols_str = ", ".join([f"'{col}'" for col in existing_cols if col not in ["id", "name"]])

        data_ref = data_ref.sem_extract_features(
            nl_prompt=(
                "YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. "
                "Use ONLY regex and rule-based approaches. "
                "DO NOT USE ANY Transformer, NER, LLM fallbacks, or LM models.\n\n"
                "Multilingual storage device titles for entity blocking.\n\n"
                f"Existing columns: {existing_cols_str}.\n\n"
                "Suggest additional features in >10% of rows: form factor, read/write speed, adapter. "
                "Lowercase strings, '0' for missing."
            ),
            input_columns=["name", "normalized_name"],  # Use both original and normalized for context
            name="discover_additional_features",
            output_columns={},  # Empty dict lets LLM discover features
            generate_via_code=True,
        )

    def fix_up_with_discovered(df):
        required_cols = [
            "id",
            "brand",
            "capacity",
            "normalized_name",
            "type",
            "model",
            "model_long",
            "model_short",
            "features",
        ]

        discovered_cols = [col for col in df.columns if col not in required_cols and col not in ["name", "id"]]
        all_cols = required_cols + discovered_cols

        if discovered_cols:
            df["_discovered_cols"] = [discovered_cols] * len(df)

        for col in all_cols:
            if col not in df.columns:
                df[col] = "0"

        for col in all_cols:
            if col != "id":
                df[col] = df[col].fillna("0").astype(str).str.lower()

        return df

    data_ref = data_ref.skb.apply_func(fix_up_with_discovered)
    result_df = data_ref.skb.eval()

    return result_df


def block_with_attr(X, attr, X2_features=None):
    if X2_features is not None and attr == "name":
        feature_cols = [col for col in X2_features.columns if col not in ["id", "name"]]
        if len(X2_features) == len(X):
            for col in feature_cols:
                if col in X2_features.columns:
                    col_values = X2_features[col].values
                    flattened_values = []
                    for val in col_values:
                        if isinstance(val, pd.Series):
                            if len(val) > 0:
                                first_val = val.iloc[0]
                                flattened_values.append(first_val if pd.notna(first_val) else None)
                            else:
                                flattened_values.append(None)
                        elif isinstance(val, (list, tuple)):
                            flattened_values.append(val[0] if len(val) > 0 else None)
                        elif hasattr(val, "__len__") and not isinstance(val, str):
                            try:
                                flattened_values.append(val[0] if len(val) > 0 else None)
                            except (TypeError, IndexError):
                                flattened_values.append(None if pd.isna(val) else val)
                        elif pd.isna(val):
                            flattened_values.append(None)
                        else:
                            flattened_values.append(val)
                    X[f"x2_{col}"] = flattened_values

            known_cols = [
                "brand",
                "capacity",
                "normalized_name",
                "mem_type",
                "type",
                "model",
                "model_long",
                "model_short",
                "item_code",
                "series",
                "pat_hb",
                "hybrid",
                "long_num",
                "features",
                "speed_rw",
                "class_info",
                "tv_phone",
                "color",
                "adapter",
                "variant",
                "generation",
                "brand_series",
            ]
        discovered_cols = [col for col in feature_cols if col not in known_cols]
        if discovered_cols:
            X["_discovered_cols"] = [discovered_cols] * len(X)

    X = X.reset_index(drop=True)

    pattern2id_1 = defaultdict(list)
    pattern2id_2 = defaultdict(list)

    for i in tqdm(range(X.shape[0])):
        if attr == "name":
            attr_i = str(X["x2_normalized_name"].iloc[i])
            pattern_1 = attr_i.lower()
            pattern2id_1[" ".join(sorted(pattern_1.split()))].append(i)
            pattern2id_1[pattern_1].append(i)

            pattern_2 = re.findall(r"\w+\s\w+\d+", attr_i)
            if len(pattern_2) != 0:
                pattern_2 = list(sorted(pattern_2))
                pattern_2 = [str(it).lower() for it in pattern_2]
                pattern2id_2[" ".join(pattern_2)].append(i)

            pattern_3 = re.findall(r"\w+\d+|\d+\w+", attr_i.lower())
            if len(pattern_3) > 0:
                pattern2id_2[" ".join(sorted(set(pattern_3)))].append(i)

            if "x2_brand" in X.columns:
                x2_brand = str(X["x2_brand"].iloc[i]) if pd.notna(X["x2_brand"].iloc[i]) else ""
                x2_capacity = (
                    str(X["x2_capacity"].iloc[i])
                    if "x2_capacity" in X.columns and pd.notna(X["x2_capacity"].iloc[i])
                    else ""
                )
                x2_type = str(X["x2_type"].iloc[i]) if "x2_type" in X.columns and pd.notna(X["x2_type"].iloc[i]) else ""
                x2_model = (
                    str(X["x2_model"].iloc[i]) if "x2_model" in X.columns and pd.notna(X["x2_model"].iloc[i]) else ""
                )
                x2_model_long = (
                    str(X["x2_model_long"].iloc[i])
                    if "x2_model_long" in X.columns and pd.notna(X["x2_model_long"].iloc[i])
                    else ""
                )
                x2_model_short = (
                    str(X["x2_model_short"].iloc[i])
                    if "x2_model_short" in X.columns and pd.notna(X["x2_model_short"].iloc[i])
                    else ""
                )
                x2_features = (
                    str(X["x2_features"].iloc[i])
                    if "x2_features" in X.columns and pd.notna(X["x2_features"].iloc[i])
                    else ""
                )
                x2_item_code = (
                    str(X["x2_item_code"].iloc[i])
                    if "x2_item_code" in X.columns and pd.notna(X["x2_item_code"].iloc[i])
                    else ""
                )
                x2_series = (
                    str(X["x2_series"].iloc[i]) if "x2_series" in X.columns and pd.notna(X["x2_series"].iloc[i]) else ""
                )
                x2_pat_hb = (
                    str(X["x2_pat_hb"].iloc[i]) if "x2_pat_hb" in X.columns and pd.notna(X["x2_pat_hb"].iloc[i]) else ""
                )
                x2_hybrid = (
                    str(X["x2_hybrid"].iloc[i]) if "x2_hybrid" in X.columns and pd.notna(X["x2_hybrid"].iloc[i]) else ""
                )
                x2_long_num = (
                    str(X["x2_long_num"].iloc[i])
                    if "x2_long_num" in X.columns and pd.notna(X["x2_long_num"].iloc[i])
                    else ""
                )
            else:
                x2_brand = ""
                x2_capacity = ""
                x2_type = ""
                x2_model = ""
                x2_model_long = ""
                x2_model_short = ""
                x2_features = ""
                x2_item_code = ""
                x2_series = ""
                x2_pat_hb = ""
                x2_hybrid = ""
                x2_long_num = ""

            if x2_features != "" and x2_features != "0":
                pattern2id_2[" ".join([x2_brand, x2_features])].append(i)

            if x2_model_long != "" and x2_model_long != "0":
                pattern2id_2[x2_model_long].append(i)
                pattern2id_2[x2_model_long.lower()].append(i)
            if x2_model_short != "" and x2_model_short != "0":
                pattern2id_2[x2_model_short].append(i)
                pattern2id_2[x2_model_short.lower()].append(i)

            if x2_brand != "" and x2_brand != "0" and x2_capacity != "" and x2_capacity != "0":
                pattern2id_2[" ".join([x2_brand, x2_capacity])].append(i)
            if (
                x2_brand != ""
                and x2_brand != "0"
                and x2_capacity != ""
                and x2_capacity != "0"
                and x2_type != ""
                and x2_type != "0"
            ):
                pattern2id_2[" ".join([x2_brand, x2_capacity, x2_type])].append(i)
            # Brand + model
            if x2_brand != "" and x2_brand != "0" and x2_model != "" and x2_model != "0":
                pattern2id_2[" ".join([x2_brand, x2_model])].append(i)

            if x2_brand != "" and x2_brand != "0":
                pattern2id_2[x2_brand].append(i)
            if x2_capacity != "" and x2_capacity != "0":
                pattern2id_2[x2_capacity].append(i)
            if x2_model != "" and x2_model != "0":
                pattern2id_2[x2_model].append(i)
            if x2_type != "" and x2_type != "0":
                pattern2id_2[x2_type].append(i)
            if x2_brand != "" and x2_brand != "0" and x2_type != "" and x2_type != "0":
                pattern2id_2[" ".join([x2_brand, x2_type])].append(i)
            if x2_model != "" and x2_model != "0" and x2_capacity != "" and x2_capacity != "0":
                pattern2id_2[" ".join([x2_model, x2_capacity])].append(i)

            if (x2_hybrid != "" and x2_hybrid != "0") or (x2_long_num != "" and x2_long_num != "0"):
                pattern2id_2[x2_hybrid + x2_long_num].append(i)

            if x2_item_code != "" and x2_item_code != "0":
                pattern2id_2[x2_item_code].append(i)

            if x2_series != "" and x2_series != "0":
                pattern2id_2[x2_series].append(i)
                if x2_brand != "" and x2_brand != "0":
                    pattern2id_2[" ".join([x2_brand, x2_series])].append(i)

            # Pat_hb pattern (hyphenated patterns)
            if x2_pat_hb != "" and x2_pat_hb != "0":
                pattern2id_2[x2_pat_hb].append(i)

            # Brand + series + capacity (strong pattern)
            if (
                x2_brand != ""
                and x2_brand != "0"
                and x2_series != ""
                and x2_series != "0"
                and x2_capacity != ""
                and x2_capacity != "0"
            ):
                pattern2id_2[" ".join([x2_brand, x2_series, x2_capacity])].append(i)

            # Use discovered features for additional blocking patterns (if available)
            if "_discovered_cols" in X.columns:
                discovered_cols = (
                    X["_discovered_cols"].iloc[0] if isinstance(X["_discovered_cols"].iloc[0], list) else []
                )
                for disc_col in discovered_cols:
                    col_name = f"x2_{disc_col}"
                    if col_name in X.columns:
                        # Use .iloc to ensure we get a scalar value, not an array
                        disc_val = X[col_name].iloc[i]
                        # Handle case where value might be an array/list/Series
                        # If it's array-like, extract the first element
                        if isinstance(disc_val, pd.Series):
                            disc_val = disc_val.iloc[0] if len(disc_val) > 0 else None
                        elif isinstance(disc_val, list):
                            disc_val = disc_val[0] if len(disc_val) > 0 else None
                        elif hasattr(disc_val, "__len__") and not isinstance(disc_val, str):
                            # Handle numpy arrays or other array-like objects
                            disc_val = disc_val[0] if len(disc_val) > 0 else None
                        # Now check if it's not NA (this should work on scalar values)
                        disc_value = str(disc_val) if pd.notna(disc_val) else ""
                        # Only use if value is meaningful and not too common
                        if disc_value != "0" and disc_value != "" and disc_value != "nan" and len(disc_value) > 2:
                            # Combine with brand for stronger blocking key
                            if x2_brand != "0":
                                pattern2id_2[" ".join([x2_brand, disc_value])].append(i)
                                # Also combine with brand+capacity for even stronger key
                                if x2_capacity != "0":
                                    pattern2id_2[" ".join([x2_brand, x2_capacity, disc_value])].append(i)

    # RELAXED thresholds for better recall
    len_threshold = 200  # Increased from 100 to allow more patterns
    if attr == "name":
        len_threshold = 300  # Increased from 150 to allow more patterns
    # add id pairs that share the same pattern to candidate set
    candidate_pairs_1 = []
    for pattern in tqdm(pattern2id_1):
        ids = list(sorted(pattern2id_1[pattern]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidate_pairs_1.append((ids[i], ids[j]))  #
    # add id pairs that share the same pattern to candidate set
    candidate_pairs_2 = []
    for pattern in tqdm(pattern2id_2):
        ids = list(sorted(pattern2id_2[pattern]))
        if len(ids) < len_threshold:  # skip patterns that are too common
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_2.append((ids[i], ids[j]))

    # remove duplicate pairs and take union
    candidate_pairs = set(candidate_pairs_2)
    candidate_pairs = candidate_pairs.union(set(candidate_pairs_1))
    candidate_pairs = list(candidate_pairs)
    jaccard_similarities = []
    candidate_pairs_real_ids = []

    if attr == "name":
        for it in tqdm(candidate_pairs):
            id1, id2 = it

            # get real ids
            real_id1 = X["id"].iloc[id1]
            real_id2 = X["id"].iloc[id2]
            if (
                real_id1 < real_id2
            ):  # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2), we only include (id1,id2) but not (id2, id1)
                candidate_pairs_real_ids.append((real_id1, real_id2))
            else:
                candidate_pairs_real_ids.append((real_id2, real_id1))

            # compute jaccard similarity
            name1 = str(X["x2_normalized_name"].iloc[id1])
            name2 = str(X["x2_normalized_name"].iloc[id2])
            s1 = set(name1.lower().split())
            s2 = set(name2.lower().split())
            denom = min(len(s1), len(s2))
            jaccard_similarities.append(len(s1.intersection(s2)) / denom if denom else 0.0)

    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    print("FINAL ", len(candidate_pairs_real_ids))
    return candidate_pairs_real_ids


def save_output(
    X1_candidate_pairs, X2_candidate_pairs
):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output_sempipes_lightweight.csv", index=False)


OUTPUT_PATH = "output_sempipes_lightweight.csv"


def save_output_X1_from_file(X1_candidate_pairs_df, X2_candidate_pairs, output_path=OUTPUT_PATH):
    expected_cand_size_X2 = 2000000

    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    X2_candidate_pairs_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])

    output_df = pd.concat([X1_candidate_pairs_df, X2_candidate_pairs_df], ignore_index=True)
    output_df.to_csv(output_path, index=False)


def run_X2():
    mode = 0
    if mode == 0:
        # X1 = pd.read_csv("experiments/sigmod/hidden_data/Z1.csv")
        X2 = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv")
    else:
        # X1 = pd.read_csv("experiments/sigmod/data/X1.csv")
        X2 = pd.read_csv("experiments/sigmod/data/X2.csv")

    # extract features for X2 using sempipes
    print("Extracting features for X2 dataset using sempipes...")
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
    )
    X2_features = extract_x2_features_sempipes(X2)
    print("Feature extraction complete.")

    # perform blocking
    X2_candidate_pairs = block_with_attr(X2, attr="name", X2_features=X2_features)

    # save results
    X1_candidate_pairs = pd.read_csv("experiments/sigmod/hidden_data/output_X1.csv")
    save_output_X1_from_file(X1_candidate_pairs, X2_candidate_pairs)


def main():
    all_recalls = []
    recalls = []
    output_path = OUTPUT_PATH
    base_path = "experiments/sigmod/hidden_data"
    input_files = ["Y1.csv", "Y2.csv"]
    nreps = 5

    for i in range(nreps):
        run_X2()
        for i, eval_dataset in enumerate(input_files):
            evaluation_dataset_path = os.path.join(base_path, eval_dataset)

            evaluation_dataset, submission_dataset = get_evaluation_dataset_with_predicted_label(
                evaluation_dataset_path, output_path, dataset_id=i + 1
            )

            # Evaluate the submission
            recall, tp, all = calculate_metrics(evaluation_dataset, submission_dataset)
            print(f"Recall for {eval_dataset} is {recall}.")
            recalls.append(recall)

        final_recall = round(np.mean(recalls), 3)
        print(f"Final recall is {final_recall}.")
        all_recalls.append(final_recall)

    print(f"Average recall is {np.mean(all_recalls)}.")
    print(f"Standard deviation is {np.std(all_recalls)}.")


if __name__ == "__main__":
    main()
