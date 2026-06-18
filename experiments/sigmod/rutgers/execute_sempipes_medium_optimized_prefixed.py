import os
import re

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator
from skrub._data_ops._evaluation import find_node_by_name

import sempipes
from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label
from experiments.sigmod.rutgers.execute_sempipes_elaborate import block_with_attr, save_output_X1_from_file
from sempipes.optimisers import EvolutionarySearch, optimise_colopro

x1_clean_pattern_1 = r"quality|new|good|best|kids|product[s]*|(?<=\s)buy\s|computer[s]*|\s[-]|(?<=i[357])-|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s](?=[m23][.\s])|(?<=[a-z])[\s]+gb|(?<=gen)[\s_](?=[134\s][0]*)"

x1_aliases = {
    "panasonic": ["pansonic"],
    "notebook": ["notebooks"],
    "tablet": ["tablets"],
    "pavilion": ["pavillion"],
    "duo ": ["core2duo ", "core 2 "],
    "hp": ["hewlett-packard"],
    "used ": ["use "],
    " ": ["cheapest", "cheap", "portable", "laptop", "kids", ";"],
}

x2_clean_pattern_1 = r"&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s][m]*(?=[23][\.\s])|(?<=usb)-[\w]+\s(?=[23][\.\s])|(?<=[a-z])[\s]+gb|(?<=data|jump)[t\s](?=trave|drive)|(?<=extreme|exceria)[\s](?=pro[\s]|plus)|(?<=class)[\s_](?=10|[234]\b)|(?<=gen)[\s_](?=[134\s][0]*)"
x2_class10_pattern = r"(10 class|class 10|class(?=[\w]+10\b)|cl\s10)"
x2_memory_clean_pattern = r"\b(msd|microvault|sd-karte|speicherkarte|minneskort|memóriakártya|flashgeheugenkaart|geheugenkaart|speicherkarten|memoriakartya|[-\s]+kaart|memory|memoria|memoire|mémoire|mamoria|tarjeta|carte|karta)"
x2_usb_clean_pattern = r"\b(flash[\s-]*drive|flash[\s-]*disk|pen[\s]*drive|micro-usb|usb-flashstation|usb-flash|usb-minne|usb-stick|speicherstick|flashgeheugen|flash|vault)"
x2_check_colors_pattern = r"silver|white|black|blue|purple|burgundy|red|green"
x2_speedrw_pattern = r"\b[0-9]{2,3}r[0-9]{2,3}w"


class BlockingModel_X2(BaseEstimator):
    """Blocking model for X2 dataset that uses block_with_attr function."""

    def __init__(self, size_of_output=2000000):
        self.size_of_output = size_of_output

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = X.reset_index(drop=True)
        X2_features = X.copy()

        if "name" in X.columns:
            X2_data = X[["id", "name"]].copy()
        elif "normalized_name" in X.columns:
            X2_data = X[["id"]].copy()
            X2_data["name"] = X["normalized_name"]
            X2_features["x2_normalized_name"] = X["normalized_name"]
        else:
            raise ValueError("Need 'name' or 'normalized_name' column for blocking")

        X2_data = X2_data.reset_index(drop=True)
        X2_features = X2_features.reset_index(drop=True)
        candidate_pairs = block_with_attr(X2_data, attr="name", X2_features=X2_features)

        if len(candidate_pairs) > self.size_of_output:
            candidate_pairs = candidate_pairs[: self.size_of_output]

        return candidate_pairs


def calculate_recall(estimator, X, y):
    predictions = estimator.predict(X)
    if isinstance(predictions, list):
        predicted_df = pd.DataFrame(predictions, columns=["left_instance_id", "right_instance_id"])
    elif isinstance(predictions, pd.DataFrame):
        predicted_df = predictions.copy()
    else:
        predicted_df = pd.DataFrame(predictions)
        if "left_instance_id" not in predicted_df.columns or "right_instance_id" not in predicted_df.columns:
            if len(predicted_df.columns) == 2:
                predicted_df.columns = ["left_instance_id", "right_instance_id"]
            else:
                raise ValueError(f"Unexpected prediction format: {type(predictions)}")

    if not isinstance(y, pd.DataFrame):
        ground_truth = pd.DataFrame(y)
    else:
        ground_truth = y.copy()
    if "lid" not in ground_truth.columns or "rid" not in ground_truth.columns:
        raise ValueError(f"Ground truth labels must have 'lid' and 'rid' columns. Got: {ground_truth.columns.tolist()}")

    predicted_df["left_right"] = predicted_df["left_instance_id"].astype(str) + predicted_df[
        "right_instance_id"
    ].astype(str)
    predicted_values = predicted_df["left_right"].values

    ground_truth["left_right"] = ground_truth["lid"].astype(str) + ground_truth["rid"].astype(str)
    reference_values = ground_truth["left_right"].values

    inter = set.intersection(set(predicted_values), set(reference_values))
    recall = len(inter) / len(reference_values) if len(reference_values) > 0 else 0.0

    return round(recall, 3)


def extract_x2_features_sempipes_dataop(
    name1: str = "extract_x2_features", name2: str = "extract_x2_features_fixed"
) -> skrub.DataOp:
    """
    Extract features from X2 dataset using sempipes - returns DataOp for optimization.
    This is the full pipeline from execute_sempipes.py but returns DataOp (not DataFrame).

    Args:
        X2_data: Input DataFrame with product data
        name: Name of the operator (used for optimization state management)
        optimize_operator: If True, this operator can be optimized. If False, it's fixed.

    Returns:
        DataOp with extracted features (not evaluated, for optimization)
    """
    data_ref = skrub.var("data_original_x2").skb.mark_as_X()
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
            "A canonicalized form of the product title for entity blocking — consistent regardless of language, encoding, or listing noise. "
            "The input is a dirty, multilingual e-commerce product title containing accented characters "
            "(e.g., e-acute, u-umlaut), HTML entities, seller prefixes, and inconsistent spacing. "
            "Non-ASCII characters transliterate to their ASCII equivalents. "
            "The same concept appears in many spellings across European languages: "
            "'classe'/'clase'/'clas '/'klasse'/'cl ' are all 'class' (speed class in French/Spanish/German); "
            "'uhs1'/'uhs-i'/'ultra high-speed' -> 'uhsi'; "
            "'typec'/'type c'/'usb-c'/'usbc' -> 'type-c'; "
            "'10 class'/'class 10'/'cl 10' -> 'class10'; 'class 4' -> 'class4'; 'class 3' -> 'class3'. "
            "Brand-specific shorthands refer to the same product: "
            "'hyperx'/'savage' -> 'kingston hxs'; 'serie ux' -> 'sony g1ux'; "
            "'dtig4'/' 101 '/'dt101g2' -> ' kingston dt101 '; 'sda10'/'sda3' -> ' kingston ultimate '; "
            "'extrem ' -> 'extreme '; 'attach' -> 'att4'. "
            "HTML entities (&amp; &reg; &nbsp;) and punctuation noise (| ; : / , + ( ) [ ]) are absent from the result. "
            "Output is lowercase with normalized whitespace."
        ),
        "brand": (
            f"The storage device manufacturer. Known brands: {brands_list}. "
            "Brand names in multilingual European listings often appear with typos, extra spaces, or hyphens: "
            "'san disk'/'san-disk'/'sandisc'/'san-disc' are all 'sandisk'; "
            "'samsun' is 'samsung'; 'kingstn' is 'kingston'; "
            "'toshbia' is 'toshiba'; 'transcent' is 'transcend'. "
            "Lowercase. '0' if no brand is identifiable."
        ),
        "capacity": (
            "Storage capacity — a strong product identity signal. "
            "Multilingual listings use language-specific unit names: French 'go'/'gigaoctet' = 'gb'; French 'to' = 'tb'; "
            "bare 'g' or 't' immediately after a number are also capacity units. "
            "Canonical form: number directly followed by unit, no space (e.g., '32gb', '1tb'). "
            "Example outputs: '32gb', '64gb 128gb', '1tb', '256gb'. "
            "'0' if no capacity is present."
        ),
        "mem_type": (
            "The primary product category. Each category has characteristic vocabulary in the title: "
            "phones contain 'lte'; XQD cards contain 'xqd'; solid-state drives contain 'ssd'; "
            "Samsung smart TVs contain 'tv'; USB drives contain 'fdrive'; memory cards contain 'memcard'. "
            "Single lowercase token from: phone, xqd, ssd, tv, fdrive, memcard. '0' if none applies."
        ),
        "type": (
            f"Product line or family name. Brand families: {families_dict}. "
            f"Intenso-specific lines: {intenso_type}. Samsung color names: {colors}. "
            "Typos are present: 'cruizer' -> 'cruzer'. "
            "Examples: 'sandisk extreme pro 32gb' -> 'extreme'; 'kingston datatraveler' -> 'datatraveler'; "
            "'samsung galaxy midnight black' -> 'midnight black'; 'intenso premium line' -> 'premium'. "
            "'0' if no product line is identifiable."
        ),
        "model": (
            "Concise model identifiers that anchor product identity across listings. "
            "These are alphanumeric codes, hyphenated codes, or letter+digit combinations at least 3 characters long. "
            "Examples: 'kingston dt101g2' -> 'dt101g2'; 'toshiba u202' -> 'u202'; 'samsung evo plus' -> 'evo plus'. "
            "'0' if no model identifier is present."
        ),
        "model_long": (
            "Manufacturer SKU or part-number codes — the strongest matching signal for entity blocking. "
            "They appear in two surface forms: "
            "lowercase hyphenated codes like 'sdsqxa-128g-anc', 'sdsdunc-032g-gn6ia', 'sdcz50-064g-b35', "
            "'thn-m401s0160e2', 'ljds45-128abst', 'lsdmi64gcbnl300', 'usm32gqx', 'mk123456', 'sr-g1ux2t'; "
            "and uppercase equivalents like 'SDCZ50-064G-B35', 'MB-MG32DA/EU', 'SDSQUNC-032G-GN6IA'. "
            "'0' if none present."
        ),
        "model_short": (
            "Brief alphanumeric model codes characteristic of storage products: "
            "'c20m', 'c20c', 'sda10', 'sda3', 'g1ux', 's720', 's705', 'u202', 'n202', 'm202', "
            "'u302', 'p20', 'g4', 'dt101', 'se9', 'a32', 's32', 'm32'. "
            "'0' if none present."
        ),
        "features": (
            "Technical specification tokens in canonical form. "
            "The same spec appears in many surface forms across European languages: "
            "USB 2.0/usb2/usb 2 -> 'usb2'; USB 3.0/usb3/usb 3 -> 'usb3'; "
            "Type-C/type c/usb-c -> 'type-c'; "
            "UHS-I/uhs-i/uhsi/uhs1 -> 'uhsi'; UHS-II -> 'uhsii'; "
            "Class 10/class10/c10 -> 'class10'; Class 4 -> 'class4'; "
            "Gen 1/gen1 -> 'gen1'; Gen 2/gen2 -> 'gen2'; "
            "SDHC -> 'sdhc'; SDXC -> 'sdxc'; OTG -> 'otg'; LTE -> 'lte'; "
            "speed patterns like '95r80w' (read/write MB/s) or '150mbs'. "
            "'0' if no technical features are present."
        ),
        "item_code": (
            "Numeric catalog or item codes enclosed in parentheses — a strong identity signal when present. "
            "Examples: '(4187407)' -> '4187407'; '(mk483394661)' -> '483394661'; '(173473)' -> '173473'. "
            "Non-numeric prefixes like 'mk' are not part of the code. "
            "When multiple codes appear, the longer one is more likely a manufacturer part number. "
            "'0' if none present."
        ),
        "series": (
            f"Product series token — brands organize storage products into named series visible in the title. "
            f"Brand family reference: {families_dict}. Intenso series: {intenso_type}. Samsung color names: {colors}. "
            "Typos are present: 'cruizer' -> 'cruzer'. "
            "Examples: 'glide', 'cruzer', 'ultimate', 'exceria', 'jumpdrive', 'premium', 'basic', 'evo', 'datatraveler'. "
            "Single lowercase token. '0' if no series is identifiable."
        ),
        "pat_hb": (
            "First hyphenated alphanumeric pattern in the title — a structural fingerprint of the product. "
            "Examples: 'uhs-i', 'type-c', 'high-speed', 'class-10', 'micro-sd', 'usb-3'. "
            "Lowercase. '0' if none present."
        ),
        "hybrid": (
            "Manufacturer product codes where letters and digits are interleaved, at least 5 characters long. "
            "These are strong product identity signals. "
            "Examples: 'dt101g2', 'usm32gqx', 'lsd16gcrbeu1000', 'sda10'. "
            "'0' if none present."
        ),
        "long_num": (
            "Numeric sequences of 4 or more consecutive digits — catalog or part codes in product titles. "
            "Examples: '4187407', '483394661', '173473'. "
            "'0' if none present."
        ),
    }

    data_ref = data_ref.sem_extract_features(
        nl_prompt=(
            "YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. "
            "Use ONLY regex and rule-based approaches. "
            "DO NOT USE ANY Transformer, NER, LLM fallbacks, or LM models.\n\n"
            "Product titles from European e-commerce sites in English, French, German, Spanish, Italian, and Polish, "
            "for removable storage devices (USB sticks, SD cards, SSDs). "
            "The same product may appear across multiple countries with different encoding (accented chars, HTML noise), "
            "language-specific unit names, and marketing variation across sellers. "
            "Features should be stable across these surface differences: "
            "two listings for the same physical product should share identical feature values.\n\n"
            "Missing a feature is worse than extracting an imperfect one. "
            "Absent features are represented as '0' (the string). All values are lowercase."
        ),
        input_columns=["name"],
        name=name2,  # Fixed name - this operator is NOT optimized
        output_columns=output_columns,
        generate_via_code=True,
    )

    def fix_up(df):
        df = df.copy()
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
                if col not in df.columns:
                    col_data = pd.Series(["0"] * len(df), index=df.index)
                else:
                    col_data = df[col]
                    if isinstance(col_data, pd.DataFrame):
                        col_data = col_data.squeeze()
                        if isinstance(col_data, pd.DataFrame):
                            col_data = col_data.iloc[:, 0]
                    if not isinstance(col_data, pd.Series):
                        col_data = pd.Series(col_data, index=df.index)

                col_series = col_data.fillna("0").astype(str)
                lower_result = col_series.str.lower()
                if not isinstance(lower_result, pd.Series):
                    lower_result = pd.Series(lower_result, index=df.index)

                df.loc[:, col] = lower_result

        df = df.copy()
        for col in df.columns:
            if col == "id":
                continue
            if not pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df

    data_ref = data_ref.skb.apply_func(fix_up)

    print("Generating additional helpful features using...")
    data_ref = data_ref.sem_gen_features(
        nl_prompt=(
            "YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. "
            "Use ONLY regex and rule-based approaches. "
            "DO NOT USE ANY Transformer, NER, LLM fallbacks, or LM models.\n\n"
            "Storage device titles from multilingual European e-commerce "
            "(USB sticks, SD/microSD cards, SSDs, memory cards, Samsung phones/TVs) "
            "in English, French, German, Spanish, Italian, and Polish. "
            "The task is entity blocking: the same physical product appears under different listings.\n\n"
            "Already extracted: brand, capacity, normalized_name, mem_type, type, model, "
            "model_long, model_short, features, item_code, series, pat_hb, hybrid, long_num.\n\n"
            "Blocking signals still present in many titles: "
            "SD card form-factor (microsd/microsdhc/microsdxc/sdhc/sdxc); "
            "transfer speed ratings in MB/s ('95r80w', '150mb/s'); "
            "adapter inclusion ('with adapter'/'con adaptador'/'mit adapter'/'avec adaptateur'); "
            "package quantity. "
            "All feature values are lowercase strings; '0' represents absence."
        ),
        name=name1,
        how_many=10,
    )
    print("Additional features discovered successfully.")

    return data_ref


def _pipeline(operator_name1, operator_name2):
    """
    Create a pipeline for X2 feature extraction and blocking.
    Uses dummy_y to handle shape mismatch between X and y.

    Args:
        X: Input DataFrame
        y: Labels (not used directly, but needed for pipeline structure)
        operator_name: Name of the operator to optimize (should be "discover_additional_blocking_features")
    """
    dummy_y = skrub.var("dummy_y").skb.mark_as_y()
    features = extract_x2_features_sempipes_dataop(operator_name1, operator_name2)

    # Apply blocking model
    return features.skb.apply(BlockingModel_X2(2000000), y=dummy_y)


def _create_env(X, y, operator_name, operator_name2, state):
    """Create environment dictionary for learner."""
    dummy_y = pd.Series([0] * len(X))
    return {
        "_skrub_X": X,
        "_skrub_y": dummy_y,
        "dummy_y": dummy_y,
        "data_original_x2": X,
        f"sempipes_memory__{operator_name}": None,
        f"sempipes_pipeline_summary__{operator_name}": None,
        f"sempipes_prefitted_state__{operator_name}": state,
        f"sempipes_inspirations__{operator_name}": None,
        f"sempipes_memory__{operator_name2}": None,
        f"sempipes_pipeline_summary__{operator_name2}": None,
        f"sempipes_prefitted_state__{operator_name2}": None,
        f"sempipes_inspirations__{operator_name2}": None,
    }


def run_X2_optimized(mode):
    """Run X2 with colopro optimization."""
    if mode == 0:
        X2 = pd.read_csv("experiments/sigmod/data/X2.csv")
        base_path_small = "experiments/sigmod/data"
        base_path_hidden = "experiments/sigmod/hidden_data"
    else:
        X2 = pd.read_csv("experiments/sigmod/data/X2.csv")
        base_path_small = "experiments/sigmod/data"
        base_path_hidden = "experiments/sigmod/data"

    sample_labels = pd.read_csv(base_path_small + "/Y2.csv")
    train_X = X2.copy()
    train_labels = sample_labels.copy()

    if mode == 0:
        test_data = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv")
        test_labels = pd.read_csv(base_path_hidden + "/Y2.csv")
    else:
        test_data = X2.copy()
        test_labels = sample_labels.copy()

    train_X["name"] = train_X["name"].str.lower()
    test_data["name"] = test_data["name"].str.lower()

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

    operator_name = "discover_additional_blocking_features"
    operator_name2 = "extract_x2_features_fixed"

    print("Pre-fitting sem_extract_features to get its fixed state...")
    temp_pipeline = extract_x2_features_sempipes_dataop(operator_name, operator_name2)
    temp_env = {
        "_skrub_X": train_X,
        "_skrub_y": pd.Series([0] * len(train_X)),
        "dummy_y": pd.Series([0] * len(train_X)),
        "data_original_x2": train_X,
        f"sempipes_prefitted_state__{operator_name2}": None,
        f"sempipes_memory__{operator_name2}": None,
        f"sempipes_pipeline_summary__{operator_name2}": None,
        f"sempipes_inspirations__{operator_name2}": None,
        f"sempipes_prefitted_state__{operator_name}": None,
        f"sempipes_memory__{operator_name}": None,
        f"sempipes_pipeline_summary__{operator_name}": None,
        f"sempipes_inspirations__{operator_name}": [],
    }
    temp_learner = temp_pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    temp_learner.fit(temp_env)

    extract_op_node = find_node_by_name(temp_pipeline, operator_name2)
    if extract_op_node is None:
        raise ValueError(f"Could not find operator node with name '{operator_name2}' in pipeline")

    extract_op_node.skb.eval(temp_env)
    if not hasattr(extract_op_node._skrub_impl, "estimator_"):
        raise ValueError(
            f"Operator node '{operator_name2}' does not have a fitted estimator. Make sure fit() was called."
        )

    fitted_estimator = extract_op_node._skrub_impl.estimator_
    fixed_extract_state = fitted_estimator.state_after_fit()
    print(f"Pre-fitted sem_extract_features state obtained. State keys: {list(fixed_extract_state.keys())}")

    pipeline_to_optimise = _pipeline(operator_name, operator_name2)

    def recall_scorer_with_labels(estimator, X_test, y=None, **kwargs):
        if isinstance(X_test, dict):
            X_test_data = X_test.get("_skrub_X", X_test)
        else:
            X_test_data = X_test

        if "id" in X_test_data.columns:
            test_ids = set(X_test_data["id"].values)
            test_labels_filtered = train_labels[
                train_labels["lid"].isin(test_ids) & train_labels["rid"].isin(test_ids)
            ].copy()
            return calculate_recall(estimator, X_test, y=test_labels_filtered, **kwargs)
        else:
            return calculate_recall(estimator, X_test, y=train_labels, **kwargs)

    print("Starting colopro optimization...")
    # Use the pre-fitted state for sem_extract_features in all evaluations
    outcomes = optimise_colopro(
        pipeline_to_optimise,
        scoring=recall_scorer_with_labels,
        cv=2,
        num_trials=36,
        search=EvolutionarySearch(population_size=6),
        additional_env_variables={
            "data_original_x2": train_X,
            "dummy_y": pd.Series([0] * len(train_X)),
            f"sempipes_prefitted_state__{operator_name2}": fixed_extract_state,  # Use pre-fitted state
        },
        n_jobs_for_evaluation=-1,
        run_name="sigmod_rutgers_sempipes_3_optimized",
        optimize_all_operators=False,

    )

    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
    print(f"Best outcome score after optimization on train CV: {best_outcome.score}, state: {best_outcome.states}")

    # Use optimized state for final prediction
    # Create environment with both the optimized state for sem_gen_features and the fixed state for sem_extract_features
    def _create_env_with_fixed_extract(X, y, operator_name, operator_name2, gen_features_state, extract_features_state):
        env = _create_env(X, y, operator_name, operator_name2, gen_features_state)
        env[f"sempipes_prefitted_state__{operator_name2}"] = extract_features_state
        return env

    learner_optimized = pipeline_to_optimise.skb.make_learner(fitted=False, keep_subsampling=False)
    learner_optimized.fit(
        _create_env_with_fixed_extract(
            train_X, train_labels, operator_name, operator_name2, best_outcome.states, fixed_extract_state
        )
    )
    optimized_results = learner_optimized.predict(
        _create_env_with_fixed_extract(
            test_data, test_labels, operator_name, operator_name2, best_outcome.states, fixed_extract_state
        )
    )

    X1_candidate_pairs = pd.read_csv("experiments/sigmod/hidden_data/output_X1.csv")
    if isinstance(optimized_results, list):
        X2_candidate_pairs = optimized_results
    else:
        X2_candidate_pairs = optimized_results

    save_output_X1_from_file(X1_candidate_pairs, X2_candidate_pairs)

    return optimized_results


def main():
    all_recalls = []
    recalls = []
    output_path = "output.csv"
    mode = 0  # 0 hidden, 1 small
    input_files = ["Y1.csv", "Y2.csv"]
    nreps = 5
    base_path = "experiments/sigmod/data" if mode == 1 else "experiments/sigmod/hidden_data"

    for i in range(nreps):
        run_X2_optimized(mode)
        for j, eval_dataset in enumerate(input_files):
            evaluation_dataset_path = os.path.join(base_path, eval_dataset)

            evaluation_dataset, submission_dataset = get_evaluation_dataset_with_predicted_label(
                evaluation_dataset_path, output_path, dataset_id=j + 1
            )

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
