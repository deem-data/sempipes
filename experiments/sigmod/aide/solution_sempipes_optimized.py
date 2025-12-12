import csv
import os
import random
import re

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator

import sempipes
from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label
from sempipes.optimisers import EvolutionarySearch, optimise_colopro


def normalize_text(s):
    s = s.lower()
    s = re.sub("[^a-z0-9]", " ", s)
    return s.strip()


def reservoir_sample(stream, k, seed=42):
    random.seed(seed)
    res = []
    for i, item in enumerate(stream):
        if i < k:
            res.append(item)
        else:
            j = random.randrange(i + 1)
            if j < k:
                res[j] = item
    return res


def generate_z1_pairs(ids, w=3):
    n = len(ids)
    for i in range(n):
        a = ids[i]
        for j in range(1, w + 1):
            if i + j < n:
                b = ids[i + j]
                yield (min(a, b), max(a, b))


def generate_z2_pairs(brands_ids, names_dict, w=5):
    # brands_ids: dict brand -> list of ids
    for brand, id_list in brands_ids.items():
        names = names_dict[brand]
        # sort by normalized name
        sorted_idx = sorted(range(len(id_list)), key=lambda i: names[i])
        n = len(sorted_idx)
        for idx in range(n):
            a = id_list[sorted_idx[idx]]
            for j in range(1, w + 1):
                if idx + j < n:
                    b = id_list[sorted_idx[idx + j]]
                    yield (min(a, b), max(a, b))


def extract_blocking_features_sem_extract(data_ref: skrub.DataOp, text_column: str, name: str) -> pd.DataFrame:
    """
    Extract blocking features using sem_extract_features (like rutgers solution).
    This uses structured feature extraction with specific output columns.
    """
    data_ref = data_ref.skb.set_description(
        f"This is a dataset with product/item descriptions in the '{text_column}' column. "
        "The goal is to extract structured features useful for entity blocking and matching. "
        "Items may be described differently (different languages, formats, typos) but represent the same product."
    )

    # Define output columns for structured feature extraction (similar to rutgers)
    # CRITICAL: All extraction must use ONLY regex and string operations - NO transformers
    output_columns = {
        "brand": (
            "PROHIBITED: Do NOT use transformers, LLMs, or any ML libraries. Use ONLY regex (re module) and string operations. "
            "Extract the main brand from the name. Common brands: sandisk, samsung, kingston, lexar, toshiba, sony, intenso, pny, transcend. "
            "Handle typos and variations: 'san disk' -> 'sandisk', 'samsun' -> 'samsung', 'kingstn' -> 'kingston'. "
            "Use regex: r'\\b(sandisk|samsung|kingston|lexar|toshiba|sony|intenso|pny|transcend)\\b' and string.replace() for variations. "
            "Return lowercase brand name or '0' if not found."
        ),
        "capacity": (
            "PROHIBITED: Do NOT use transformers, LLMs, or any ML libraries. Use ONLY regex (re module) and string operations. "
            "Extract storage capacity. Look for patterns like '32gb', '64gb', '128gb', '256gb', '512gb', '1tb', '2tb'. "
            "Handle multilingual units: 'gb' (English), 'go' (French), 'g' -> normalize to 'gb'. "
            "Use regex: r'([1-9]{1,3})[-\\s]*[g][bo]?' for GB, r'([1-9])[-\\s]*[t][bo]?' for TB. "
            "Normalize 'go' -> 'gb', 'g' -> 'gb', 'to' -> 'tb' using string.replace(). Return like '32gb', '128gb', '1tb' or '0'."
        ),
        "model_code": (
            "PROHIBITED: Do NOT use transformers, LLMs, or any ML libraries. Use ONLY regex (re module) and string operations. "
            "Extract model/SKU codes. Look for alphanumeric patterns with hyphens like 'SDCZ50-064G-B35', 'MB-MG32DA/EU'. "
            "Also extract short codes like 'dt101', 'sda10', 'g1ux', 'u202'. "
            "Use regex: r'([A-Z]{2,}[0-9A-Z-]+(?:[/-][0-9A-Z-]+)*)' for long codes, r'\\b([a-z]{2,3}[0-9]{2,4})\\b' for short codes. "
            "Use re.findall() to get all matches. Return space-separated codes or '0'."
        ),
        "product_type": (
            "PROHIBITED: Do NOT use transformers, LLMs, or any ML libraries. Use ONLY regex (re module) and string operations. "
            "Extract product type: 'microsd', 'sd', 'usb', 'ssd', 'cf', 'xf'. "
            "Normalize variations: 'micro sd' -> 'microsd', 'micro-sd' -> 'microsd' using string.replace(). "
            "Use regex: r'\\b(microsd|micro-sd|micro sd|sd|usb|ssd|cf|xf)\\b'. "
            "Return lowercase type or '0'."
        ),
        "speed_class": (
            "PROHIBITED: Do NOT use transformers, LLMs, or any ML libraries. Use ONLY regex (re module) and string operations. "
            "Extract speed/class ratings: 'class10', 'class4', 'uhs-i', 'uhs-ii', 'uhs1', 'uhs2', 'v30', 'v10', 'a1', 'a2'. "
            "Normalize: 'class 10' -> 'class10', 'uhs-i' -> 'uhsi', 'uhs-1' -> 'uhs1' using string.replace(). "
            "Use regex: r'\\b(class[\\s_]?[0-9]{1,2}|uhs[\\s_-]?[i12]{1,2}|v[0-9]{1,2}|a[12])\\b'. "
            "Return space-separated classes or '0'."
        ),
        "usb_version": (
            "PROHIBITED: Do NOT use transformers, LLMs, or any ML libraries. Use ONLY regex (re module) and string operations. "
            "Extract USB version: 'usb3', 'usb30', 'usb31', 'usb32', 'usbc', 'type-c'. "
            "Normalize: 'usb 3.0' -> 'usb30', 'usb-3.0' -> 'usb30', 'usb c' -> 'usbc', 'type c' -> 'typec' using string.replace(). "
            "Use regex: r'\\b(usb[\\s-]?[23][\\.\\s]?[01]?|usb[\\s-]?c|type[\\s-]?c)\\b'. "
            "Return lowercase version or '0'."
        ),
        "capacity_value": (
            "PROHIBITED: Do NOT use transformers, LLMs, or any ML libraries. Use ONLY regex (re module) and string operations. "
            "Extract just the numeric capacity value (without unit). Extract numbers like '32', '64', '128', '256', '512', '1', '2'. "
            "Use regex: r'\\b([1-9]{1,3})\\s*(?:gb|go|g|tb|to|t)\\b' to get the number before capacity units. "
            "Use re.findall() or re.search() to extract. Return the number as string (e.g., '32', '128', '1') or '0'."
        ),
        "normalized_name": (
            "PROHIBITED: Do NOT use transformers, LLMs, or any ML libraries. Use ONLY regex (re module) and string operations. "
            "Clean and normalize the product name: convert to lowercase, remove special chars except spaces, "
            "normalize whitespace, handle accented characters. This is a cleaned version of the original name. "
            "Use: str(name).lower(), re.sub(r'[^a-z0-9\\s]', ' ', name), normalize whitespace using string.replace(). "
            "Return the normalized string."
        ),
    }

    # Extract features using sem_extract_features
    data_ref = data_ref.sem_extract_features(
        nl_prompt=(
            "CRITICAL: YOU ARE ABSOLUTELY PROHIBITED FROM USING THE TRANSFORMERS LIBRARY, ANY TRANSFORMER MODELS, "
            "LANGUAGE MODELS (LMs), LARGE LANGUAGE MODELS (LLMs), OR ANY NEURAL NETWORK MODELS. "
            "You MUST use ONLY regex (re module) and basic string operations. "
            "DO NOT import or use: transformers, torch, tensorflow, keras, huggingface, sentence_transformers, "
            "or any other ML/DL libraries. "
            "DO NOT use chr() function in lambda expressions - use string.replace() for common HTML entities instead. "
            "\n\n"
            "Extract structured features from product names for entity blocking and matching. "
            "The goal is to find duplicate products that may be described differently. "
            "Focus on extracting: brand, capacity, model codes, product type, speed classes, USB versions. "
            "These features will be used to create blocking keys that help match similar products even with different names. "
            "\n\n"
            "CODE REQUIREMENTS:\n"
            "- Use ONLY regex (re module) and basic string operations (str.lower(), str.replace(), str.split(), etc.). "
            "- DO NOT use str.extract() without capture groups - use str.findall() or str.contains() instead. "
            "- Return features as lowercase strings, '0' for missing values. "
            "- Handle edge cases: missing values, empty strings, special characters. "
            "- Optimize for throughput - must handle millions of rows efficiently."
        ),
        input_columns=[text_column],
        name=name,
        output_columns=output_columns,
    )

    # Evaluate and format
    extracted = data_ref.skb.eval()

    # Ensure all extracted columns are properly formatted
    for col in extracted.columns:
        if col not in ["id", text_column]:
            extracted[col] = extracted[col].fillna("0").astype(str).str.lower()

    return extracted


class BlockingModel(BaseEstimator):
    """Blocking model that uses token-based blocking on sempipes features."""

    def __init__(self, size_of_output=2000000):
        self.size_of_output = size_of_output

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, dict):
            X_data = X.get("_skrub_X", X)
        else:
            X_data = X

        if not isinstance(X_data, pd.DataFrame):
            X_data = pd.DataFrame(X_data)

        X_data = X_data.reset_index(drop=True)

        # Extract original columns and features
        text_column = "name"
        if text_column not in X_data.columns:
            # Try to find a name-like column
            for col in ["name", "normalized_name", "title"]:
                if col in X_data.columns:
                    text_column = col
                    break

        # Prepare data
        z2 = X_data[["id"]].copy()
        if "brand" in X_data.columns:
            z2["brand"] = X_data["brand"]
        else:
            z2["brand"] = ""
        if text_column in X_data.columns:
            z2["name"] = X_data[text_column]
        else:
            z2["name"] = ""

        z2["brand_n"] = z2["brand"].fillna("").map(normalize_text)
        z2["name_n"] = z2["name"].fillna("").map(normalize_text)

        # Get feature columns - use ALL columns generated by sempipes
        exclude_cols = ["id", "brand", "name", "brand_n", "name_n"]
        feature_cols = [col for col in X_data.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if col in X_data.columns]

        if len(feature_cols) == 0:
            # Fallback to original blocking if no features
            return self._fallback_blocking(z2)

        # Merge features
        z2_with_features = z2.merge(X_data[["id"] + feature_cols], on="id", how="left")

        # Only use features that are actually generated by sempipes (not original columns)
        original_cols = ["id", "brand", "name", "brand_n", "name_n"]
        sempipes_feature_cols = [col for col in feature_cols if col not in original_cols]

        if len(sempipes_feature_cols) == 0:
            return self._fallback_blocking(z2)

        # Token-based blocking (same logic as solution_sempipes.py)
        from collections import defaultdict

        pattern2id_by_feature = {col: defaultdict(list) for col in sempipes_feature_cols}
        pattern2id_original_1 = defaultdict(list)
        pattern2id_original_2 = defaultdict(list)

        z2_with_features = z2_with_features.reset_index(drop=True)

        for i in range(len(z2_with_features)):
            id_val = z2_with_features.loc[i, "id"]

            # Use ALL sempipes features for blocking
            for col in sempipes_feature_cols:
                if col in z2_with_features.columns:
                    feature_value = str(z2_with_features.loc[i, col]).lower()
                    if feature_value and feature_value != "0":
                        tokens = feature_value.split()
                        for token in tokens:
                            if len(token) > 1:
                                pattern2id_by_feature[col][token].append(id_val)
                        if len(feature_value) < 200:
                            pattern2id_by_feature[col][feature_value].append(id_val)

            # Original blocking patterns
            name_val = str(z2_with_features.loc[i, "name_n"])
            pattern2id_original_1[name_val].append(id_val)

            pattern_2 = re.findall(r"\w+\s\w+\d+", name_val)
            if len(pattern_2) > 0:
                pattern_2 = list(sorted(pattern_2))
                pattern_2 = [str(it).lower() for it in pattern_2]
                pattern2id_original_2[" ".join(pattern_2)].append(id_val)

        # Collect candidate pairs
        candidate_pairs_set = set()

        for col in sempipes_feature_cols:
            for pattern in pattern2id_by_feature[col]:
                ids = list(sorted(pattern2id_by_feature[col][pattern]))
                threshold = 500 if len(pattern) > 10 else 200
                if len(ids) < threshold and len(ids) > 1:
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            candidate_pairs_set.add((ids[i], ids[j]))

        for pattern in pattern2id_original_1:
            ids = list(sorted(pattern2id_original_1[pattern]))
            if len(ids) < 1000 and len(ids) > 1:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        candidate_pairs_set.add((ids[i], ids[j]))

        for pattern in pattern2id_original_2:
            ids = list(sorted(pattern2id_original_2[pattern]))
            if len(ids) < 100 and len(ids) > 1:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        candidate_pairs_set.add((ids[i], ids[j]))

        candidate_pairs = list(candidate_pairs_set)

        # Rank by jaccard similarity
        jaccard_similarities = []
        candidate_pairs_real_ids = []

        id_to_idx = {z2_with_features.loc[i, "id"]: i for i in range(len(z2_with_features))}

        for pair in candidate_pairs:
            id1, id2 = pair

            if id1 < id2:
                candidate_pairs_real_ids.append((id1, id2))
            else:
                candidate_pairs_real_ids.append((id2, id1))

            idx1 = id_to_idx.get(id1)
            idx2 = id_to_idx.get(id2)

            if idx1 is None or idx2 is None:
                jaccard = 0.0
            else:
                all_tokens1 = set()
                all_tokens2 = set()
                for col in sempipes_feature_cols:
                    if col in z2_with_features.columns:
                        val1 = str(z2_with_features.loc[idx1, col]).lower()
                        val2 = str(z2_with_features.loc[idx2, col]).lower()
                        if val1 and val1 != "0":
                            all_tokens1.update(val1.split())
                        if val2 and val2 != "0":
                            all_tokens2.update(val2.split())

                if len(all_tokens1) == 0 or len(all_tokens2) == 0:
                    name1 = str(z2_with_features.loc[idx1, "name_n"])
                    name2 = str(z2_with_features.loc[idx2, "name_n"])
                    all_tokens1 = set(name1.lower().split())
                    all_tokens2 = set(name2.lower().split())

                if len(all_tokens1) > 0 or len(all_tokens2) > 0:
                    jaccard = len(all_tokens1.intersection(all_tokens2)) / max(len(all_tokens1), len(all_tokens2))
                else:
                    jaccard = 0.0

            jaccard_similarities.append(jaccard)

        # Sort by similarity
        candidate_pairs_real_ids = [
            x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)
        ]

        # Take top pairs
        cand2 = candidate_pairs_real_ids[: self.size_of_output]
        return cand2

    def _fallback_blocking(self, z2):
        """Fallback to original blocking if no features available."""
        brands = {}
        for b, grp in z2.groupby("brand_n"):
            ids = grp["id"].tolist()
            if len(ids) > 1:
                brands[b] = ids

        names_dict = {}
        for b, grp in z2.groupby("brand_n"):
            ids = grp["id"].tolist()
            nm = grp["name_n"].tolist()
            if len(ids) > 1:
                names_dict[b] = nm

        stream2 = generate_z2_pairs(brands, names_dict, w=5)
        cand2 = reservoir_sample(stream2, self.size_of_output, seed=42)
        return cand2


def calculate_recall(estimator, X, y):
    """
    Sklearn-compatible scorer function that calculates recall.
    """
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


def _pipeline(operator_name):
    """
    Create pipeline for optimization.
    This pipeline uses only sem_gen_features (optimizable).
    """
    data_ref = skrub.var("data_original_z2").skb.mark_as_X()
    y = skrub.var("dummy_y").skb.mark_as_y()

    # Apply sem_gen_features (this is the operator we optimize)
    data_ref = data_ref.skb.set_description(
        "This is a dataset with product/item descriptions in the 'name' column. "
        "The goal is to generate features useful for entity blocking and matching. "
        "Items may be described differently (different languages, formats, typos) but represent the same product. "
        "We need features that help identify matching pairs even when names differ significantly."
    )

    data_ref = data_ref.sem_gen_features(
        nl_prompt=(
            "CRITICAL TASK: Generate blocking features for entity matching. "
            "Your features will be used to find duplicate products that may have VERY DIFFERENT names. "
            "The dataset contains product names in the 'name' column. "
            "\n\nBLOCKING STRATEGY: "
            "We group items by brand, then sort by features, then generate pairs of nearby items. "
            "Your features should help items with similar characteristics (capacity, model, type) be sorted together "
            "EVEN IF their names are completely different. "
            "\n\nFEATURE GENERATION PRIORITIES (in order of importance):\n"
            "1. CAPACITY/STORAGE SIZE: Extract numeric capacity values (32, 64, 128, 256, 512, etc.) "
            "   with normalized units (gb, go, tb, to, mb, mo). This is CRITICAL - many matches differ only in name but have same capacity.\n"
            "2. MODEL/SKU CODES: Extract alphanumeric model codes, SKUs, product numbers. "
            "   Look for patterns like: 'SDXC', 'SDHC', 'UHS-I', 'UHS-II', 'Class 10', 'V30', 'A1', 'A2', etc.\n"
            "3. PRODUCT TYPE/FORMAT: Extract card type (microSD, SD, CF, etc.), format (SDXC, SDHC), interface type.\n"
            "4. SPEED/PERFORMANCE: Extract speed ratings (MB/s, Mbps, Class ratings, UHS classes, Video speed classes).\n"
            "5. USB VERSION: Extract USB specifications (USB 3.0, USB 3.1, USB-C, etc.).\n"
            "6. BRAND VARIATIONS: Normalize brand name variations (san disk -> sandisk, samsun -> samsung).\n"
            "7. KEY PHRASES: Extract important 2-3 word phrases that appear in product names.\n"
            "8. NORMALIZED TEXT: Cleaned versions of the name (lowercase, special chars removed, whitespace normalized).\n"
            "\n\nFEATURE REQUIREMENTS:\n"
            "1. Features MUST be extractable using ONLY regex (re module) and basic string operations. "
            "2. Features should be commonly present (>5% of rows) to be useful for blocking. "
            "3. Features should normalize variations (e.g., '32GB' and '32 Go' should both extract '32gb'). "
            "4. For numeric features, extract the NUMBER as a string (e.g., '32', '128', '256'). "
            "5. For categorical features, normalize to lowercase and common forms. "
            "\n\nCRITICAL FOR MAXIMUM RECALL:\n"
            "- Generate MANY features (aim for 12+ features) to catch different matching patterns. "
            "- It's MUCH better to have features that catch potential matches (even with false positives) than to miss matches. "
            "- Missing a feature means missing potential matches - be comprehensive! "
            "- Features should help items with same capacity/model/type be sorted together even with different names. "
            "\n\nCODE REQUIREMENTS - CRITICAL PROHIBITIONS:\n"
            "- ABSOLUTELY PROHIBITED: Do NOT use transformers library, transformer models, language models (LMs), "
            "large language models (LLMs), or any neural network models. "
            "- DO NOT import or use: transformers, torch, tensorflow, keras, huggingface, sentence_transformers, "
            "or any other ML/DL libraries. "
            "- Use ONLY regex (re module) and basic string operations (str.lower(), str.replace(), str.split(), etc.). "
            "- DO NOT use str.extract() without capture groups - use str.findall() or str.contains() instead. "
            "- DO NOT use chr() function in lambda expressions - use string.replace() for common HTML entities instead. "
            "- Return features as lowercase strings, '0' for missing values. "
            "- Handle edge cases: missing values, empty strings, special characters. "
            "- Optimize for throughput - must handle millions of rows efficiently. "
            "\n\nEXAMPLE FEATURES TO GENERATE:\n"
            "- Capacity in GB (normalized): '32gb', '64gb', '128gb', '256gb', '512gb', '1tb'\n"
            "- Model/SKU code: 'sdxc', 'sdhc', 'uhs-i', 'uhs-ii', 'class10', 'v30', 'a1', 'a2'\n"
            "- Card type: 'microsd', 'sd', 'cf', 'xf'\n"
            "- Speed value: '100', '150', '200' (from '100MB/s', '150Mbps', etc.)\n"
            "- USB version: 'usb3', 'usb30', 'usb31', 'usbc'\n"
            "- Brand normalized: 'sandisk', 'samsung', 'kingston'\n"
            "- Key phrases: 'ultra plus', 'extreme pro', 'professional'\n"
        ),
        name=operator_name,
        how_many=8,
    )

    # Format generated features
    def fix_up_generated(df):
        for col in df.columns:
            if col not in ["id", "name"]:
                df[col] = df[col].fillna("0").astype(str).str.lower()
        return df

    features = data_ref.skb.apply_func(fix_up_generated)
    return features.skb.apply(BlockingModel(2000000), y=y)


def _create_env(X, y, operator_name, gen_features_state):
    """Create environment dictionary for learner."""
    dummy_y = pd.Series([0] * len(X), name="dummy")

    return {
        "_skrub_X": X,
        "_skrub_y": dummy_y,
        "data_original_z2": X,
        "dummy_y": dummy_y,
        f"sempipes_memory__{operator_name}": None,
        f"sempipes_pipeline_summary__{operator_name}": None,
        f"sempipes_prefitted_state__{operator_name}": gen_features_state,
        f"sempipes_inspirations__{operator_name}": None,
    }


def save_output(X1_candidate_pairs, X2_candidate_pairs, output_path="working/submission.csv"):
    """Save the candidate set for both datasets to a SINGLE file"""
    expected_cand_size_X1 = 1_000_000
    expected_cand_size_X2 = 2_000_000

    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv(output_path, index=False)


def compute_z1_blocking():
    """Compute blocking for Z1 dataset"""
    z1 = pd.read_csv("experiments/sigmod/hidden_data/Z1.csv", usecols=["id", "title"])
    z1["title_n"] = z1["title"].fillna("").map(normalize_text)
    z1 = z1.sort_values("title_n")
    ids1 = z1["id"].tolist()
    stream1 = generate_z1_pairs(ids1, w=3)
    cand1 = reservoir_sample(stream1, 1_000_000, seed=0)
    return cand1


if __name__ == "__main__":
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-3-flash-preview",
            parameters={"temperature": 1.5},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="gemini/gemini-3-flash-preview",
            parameters={"temperature": 1.5},
        ),
    )

    print("Performing blocking for Z1 (original method)...")
    cand1 = pd.read_csv("aide_output_Z1.csv").values.tolist()

    num_repeats = 5
    print(f"Repeating blocking for Z2 {num_repeats} times with optimization...")

    all_cand2 = []
    Z2_recalls = []

    base_path = "experiments/sigmod/hidden_data"
    evaluation_dataset_path_Y2 = os.path.join(base_path, "Y2.csv")
    output_path = "experiments/sigmod/results/output_aide_sempipes_optimized.csv"

    # Load training data
    train_X = pd.read_csv("experiments/sigmod/data/X2.csv", usecols=["id", "brand", "name"])
    train_X["brand_n"] = train_X["brand"].fillna("").map(normalize_text)
    train_X["name_n"] = train_X["name"].fillna("").map(normalize_text)
    train_labels = pd.read_csv("experiments/sigmod/data/Y2.csv")

    # Load test data
    test_data = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv", usecols=["id", "brand", "name"])
    test_data["brand_n"] = test_data["brand"].fillna("").map(normalize_text)
    test_data["name_n"] = test_data["name"].fillna("").map(normalize_text)

    for repeat_idx in range(num_repeats):
        print(f"\n--- Repeat {repeat_idx + 1}/{num_repeats} ---")

        operator_name = f"generate_z2_additional_features_repeat_{repeat_idx}"

        # Create pipeline for optimization
        pipeline_to_optimise = _pipeline(operator_name)

        # Create scorer
        def recall_scorer_with_labels(estimator, X_test, y=None, **kwargs):
            if isinstance(X_test, dict):
                X_test_data = X_test.get("_skrub_X", X_test)
            else:
                X_test_data = X_test
            if "id" in X_test_data.columns:
                test_ids = set(X_test_data["id"].values)
                test_labels = train_labels[
                    train_labels["lid"].isin(test_ids) & train_labels["rid"].isin(test_ids)
                ].copy()
                return calculate_recall(estimator, X_test, y=test_labels, **kwargs)
            else:
                return calculate_recall(estimator, X_test, y=train_labels, **kwargs)

        print(f"Starting colopro optimization (repeat {repeat_idx + 1})...")
        outcomes = optimise_colopro(
            pipeline_to_optimise,
            operator_name,
            scoring=recall_scorer_with_labels,
            cv=5,
            num_trials=24,
            search=EvolutionarySearch(population_size=6),
            additional_env_variables={
                "data_original_z2": train_X,
                "dummy_y": pd.Series([0] * len(train_X), name="dummy"),
            },
        )

        best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
        print(f"Best outcome score after optimization on train CV (repeat {repeat_idx + 1}): {best_outcome.score}")

        # Use optimized state for final prediction
        learner_optimized = pipeline_to_optimise.skb.make_learner(fitted=False, keep_subsampling=False)
        learner_optimized.fit(_create_env(train_X, None, operator_name, best_outcome.state))
        cand2 = learner_optimized.predict(_create_env(test_data, None, operator_name, best_outcome.state))

        all_cand2.extend(cand2)
        print(f"Found {len(cand2)} candidate pairs in repeat {repeat_idx + 1}")

        # Save output and evaluate
        save_output(cand1, cand2, output_path)

        evaluation_dataset, submission_dataset = get_evaluation_dataset_with_predicted_label(
            evaluation_dataset_path_Y2, output_path, dataset_id=2
        )

        recall, tp, all = calculate_metrics(evaluation_dataset, submission_dataset)
        Z2_recalls.append(recall)
        print(f"Recall for Y2.csv (repeat {repeat_idx + 1}) is {recall}.")

    avg_recall = np.mean(Z2_recalls)
    std_recall = np.std(Z2_recalls)
    print(f"\nAverage recall: {round(avg_recall, 3)}")
    print(f"Standard deviation: {round(std_recall, 3)}")
