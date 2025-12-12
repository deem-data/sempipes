import csv
import os
import random
import re

import numpy as np
import pandas as pd
import skrub

import sempipes
from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label


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


def extract_blocking_features(data_ref: skrub.DataOp, text_column: str, name: str, how_many: int = 12) -> pd.DataFrame:
    """
    Generate blocking-relevant features from text using sem_gen_features.
    This automatically discovers features useful for blocking and entity matching.
    """
    data_ref = data_ref.skb.set_description(
        f"This is a dataset with product/item descriptions in the '{text_column}' column. "
        "The goal is to generate features useful for entity blocking and matching. "
        "Items may be described differently (different languages, formats, typos) but represent the same product. "
        "We need features that help identify matching pairs even when names differ significantly."
    )

    # Generate blocking features automatically using sem_gen_features
    # Increased how_many to get more diverse features for better recall
    data_ref = data_ref.sem_gen_features(
        nl_prompt=(
            "CRITICAL TASK: Generate blocking features for entity matching. "
            "Your features will be used to find duplicate products that may have VERY DIFFERENT names. "
            f"The dataset contains product names in the '{text_column}' column. "
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
        name=name,
        how_many=how_many,
    )

    # Evaluate the DataOp to get the DataFrame
    extracted = data_ref.skb.eval()

    # Ensure all generated feature columns are properly formatted
    for col in extracted.columns:
        if col not in ["id", text_column]:  # Skip id and original text column
            extracted[col] = extracted[col].fillna("0").astype(str).str.lower()

    return extracted


def generate_z2_pairs_with_features(brands_ids, id_to_features, feature_cols, w=5):
    """
    Generate pairs using multiple feature columns for better blocking.
    This function is flexible and will use ANY columns provided in feature_cols,
    regardless of their names or how they were generated.

    Optimized with aggressive limits to avoid performance issues.

    brands_ids: dict brand -> list of ids
    id_to_features: dict id -> dict of feature values (can contain any columns)
    feature_cols: list of feature column names to use for blocking (any columns sempipes generates)
    """
    all_pairs = set()

    # BALANCED LIMITS: More aggressive to get better recall, but still efficient
    max_brands_to_process = 500  # Process top 500 brands by size (increased from 200)
    max_items_per_brand = 8000  # Allow larger brand groups (increased from 5000)
    max_pairs_per_feature = 5000  # More pairs per feature (increased from 2000)
    min_items_with_feature = 2  # Lower threshold to catch more features (reduced from 3)
    max_total_pairs = 200000  # Higher limit on total pairs (increased from 50000)

    brands_sorted = sorted(brands_ids.items(), key=lambda x: len(x[1]), reverse=True)[:max_brands_to_process]

    for brand, id_list in brands_sorted:
        # Skip very large brand groups to avoid performance issues
        if len(id_list) > max_items_per_brand:
            continue

        # Early exit if we've generated enough pairs
        if len(all_pairs) >= max_total_pairs:
            break

        # Try each feature column for blocking - works with any columns sempipes generates
        for col in feature_cols:
            # Early exit if we've generated enough pairs
            if len(all_pairs) >= max_total_pairs:
                break

            # Create list of (id, feature_value) pairs for this brand
            id_feature_pairs = []
            for id_val in id_list:
                feature_val = id_to_features.get(id_val, {}).get(col, "0")
                # Skip if feature value is too common (like "0" or empty)
                if feature_val and str(feature_val).lower() not in ["0", "", "nan", "none"]:
                    id_feature_pairs.append((id_val, str(feature_val).lower()))

            # Skip if too few items have this feature
            if len(id_feature_pairs) < min_items_with_feature:
                continue

            # Sort by feature value
            id_feature_pairs.sort(key=lambda x: x[1])
            n = len(id_feature_pairs)

            # Limit pairs per feature to avoid explosion
            pairs_for_this_feature = 0

            for idx in range(n):
                if pairs_for_this_feature >= max_pairs_per_feature or len(all_pairs) >= max_total_pairs:
                    break
                a = id_feature_pairs[idx][0]
                for j in range(1, w + 1):
                    if (
                        idx + j < n
                        and pairs_for_this_feature < max_pairs_per_feature
                        and len(all_pairs) < max_total_pairs
                    ):
                        b = id_feature_pairs[idx + j][0]
                        all_pairs.add((min(a, b), max(a, b)))
                        pairs_for_this_feature += 1

    return all_pairs


def save_output(X1_candidate_pairs, X2_candidate_pairs, output_path="working/submission.csv"):
    """Save the candidate set for both datasets to a SINGLE file"""
    expected_cand_size_X1 = 1_000_000
    expected_cand_size_X2 = 2_000_000

    # Make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # Make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
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


def compute_z2_blocking_with_features(seed=1, repeat_idx=0):
    """Compute blocking for Z2 dataset using sempipes-extracted features"""
    z2 = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv", usecols=["id", "brand", "name"])
    z2["brand_n"] = z2["brand"].fillna("").map(normalize_text)
    z2["name_n"] = z2["name"].fillna("").map(normalize_text)

    # Extract blocking features using sempipes
    # Try sem_extract_features first (structured extraction like rutgers), then sem_gen_features for additional features
    print(f"Extracting blocking features for Z2 (repeat {repeat_idx + 1})...")
    z2_dataop = skrub.var("Z2", z2).skb.set_name("Z2")

    # Use sem_extract_features for structured feature extraction (like rutgers)
    print("  Using sem_extract_features for structured features...")
    z2_features_structured = extract_blocking_features_sem_extract(
        z2_dataop, "name", f"extract_z2_structured_features_repeat_{repeat_idx}"
    )

    # Also use sem_gen_features for additional discovered features
    print("  Using sem_gen_features for additional features...")
    z2_dataop2 = skrub.var("Z2", z2_features_structured).skb.set_name("Z2")
    z2_features_additional = extract_blocking_features(
        z2_dataop2,
        "name",
        f"generate_z2_additional_features_repeat_{repeat_idx}",
        how_many=8,  # Generate additional features beyond structured ones
    )

    # Merge both feature sets
    z2_features = z2_features_structured.merge(
        z2_features_additional[
            ["id"] + [col for col in z2_features_additional.columns if col not in z2_features_structured.columns]
        ],
        on="id",
        how="left",
    )

    # Get feature columns - use ALL columns generated by sempipes
    # Only exclude: 'id' (needed for merging) and the original text column 'name' (we use name_n instead)
    # This ensures we use any columns sempipes generates, regardless of their names
    exclude_cols = ["id", "name"]  # Minimal exclusion: only id and original text column
    feature_cols = [col for col in z2_features.columns if col not in exclude_cols]

    # Ensure all feature_cols actually exist in z2_features (safety check)
    feature_cols = [col for col in feature_cols if col in z2_features.columns]

    print(
        f"Using {len(feature_cols)} feature columns for blocking: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}"
    )

    # Merge features back with original data
    # Only merge columns that actually exist in z2_features to avoid KeyErrors
    cols_to_merge = ["id"] + [col for col in feature_cols if col in z2_features.columns]
    z2_with_features = z2.merge(z2_features[cols_to_merge], on="id", how="left")

    # Ensure brand_n and name_n exist (they should from z2, but verify)
    # This is important for the groupby operations later
    if "brand_n" not in z2_with_features.columns:
        z2_with_features["brand_n"] = z2["brand_n"].values
    if "name_n" not in z2_with_features.columns:
        z2_with_features["name_n"] = z2["name_n"].values

    # Build id -> features mapping, using ALL available feature columns
    # This handles any columns sempipes generates, even if we didn't expect them
    # Optimize by using to_dict - much faster than iterating
    print("Building feature mapping...")
    # Only include feature columns that actually exist
    available_feature_cols = [col for col in feature_cols if col in z2_with_features.columns]

    # Use to_dict for fast conversion - creates nested dict with id as key
    # This is much faster than iterating over rows
    feature_df = z2_with_features[["id"] + available_feature_cols].set_index("id")
    id_to_features = feature_df.to_dict("index")

    # Convert to the format we need (dict of dicts with string keys)
    id_to_features = {
        int(k): {col: str(v.get(col, "0")) for col in available_feature_cols} for k, v in id_to_features.items()
    }

    print(f"Built feature mapping for {len(id_to_features)} items")

    # Build blocking indices by brand

    brands = {}
    for b, grp in z2_with_features.groupby("brand_n"):
        ids = grp["id"].tolist()
        if len(ids) > 1:
            brands[b] = ids

    # Also use original normalized name as a feature
    names_dict = {}
    for b, grp in z2.groupby("brand_n"):
        ids = grp["id"].tolist()
        nm = grp["name_n"].tolist()
        if len(ids) > 1:
            names_dict[b] = nm

    # DRASTIC CHANGE: Make sempipes features PRIMARY blocking strategy (like baseline)
    # Use token-based blocking on ALL sempipes features - this is much more aggressive
    from collections import defaultdict

    # Only use features that are actually generated by sempipes (not original columns)
    original_cols = ["id", "brand", "name", "brand_n", "name_n"]
    sempipes_feature_cols = [col for col in feature_cols if col not in original_cols]

    print(f"Using token-based blocking on {len(sempipes_feature_cols)} sempipes features...")

    # Build token-based blocking indices (like baseline_sempipes.py)
    pattern2id_by_feature = {col: defaultdict(list) for col in sempipes_feature_cols}

    # Also keep original name-based patterns
    pattern2id_original_1 = defaultdict(list)
    pattern2id_original_2 = defaultdict(list)

    # Build indices from sempipes features (TOKEN-BASED - very aggressive)
    # Use positional indexing for efficiency
    z2_with_features = z2_with_features.reset_index(drop=True)

    # Build id -> index mapping
    id_to_idx = {}
    for i in range(len(z2_with_features)):
        id_val = z2_with_features.loc[i, "id"]
        id_to_idx[id_val] = i

        # Use ALL sempipes features for blocking
        for col in sempipes_feature_cols:
            if col in z2_with_features.columns:
                feature_value = str(z2_with_features.loc[i, col]).lower()
                if feature_value and feature_value != "0":
                    # TOKEN-BASED: Split by space and create blocks for each token
                    tokens = feature_value.split()
                    for token in tokens:
                        if len(token) > 1:  # Skip single characters
                            pattern2id_by_feature[col][token].append(id_val)
                    # Also block on the full feature value if it's not too long
                    if len(feature_value) < 200:
                        pattern2id_by_feature[col][feature_value].append(id_val)

        # Original blocking patterns (as fallback)
        name_val = str(z2_with_features.loc[i, "name_n"])
        pattern2id_original_1[name_val].append(id_val)

        pattern_2 = re.findall(r"\w+\s\w+\d+", name_val)
        if len(pattern_2) > 0:
            pattern_2 = list(sorted(pattern_2))
            pattern_2 = [str(it).lower() for it in pattern_2]
            pattern2id_original_2[" ".join(pattern_2)].append(id_val)

    # Collect candidate pairs from ALL blocking strategies
    candidate_pairs_set = set()

    # Block on each sempipes feature (TOKEN-BASED - generates many pairs)
    print("Generating pairs from sempipes features (token-based)...")
    for col in sempipes_feature_cols:
        for pattern in pattern2id_by_feature[col]:
            ids = list(sorted(pattern2id_by_feature[col][pattern]))
            # Adjust threshold - more specific patterns can have higher thresholds
            threshold = 500 if len(pattern) > 10 else 200
            if len(ids) < threshold and len(ids) > 1:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        candidate_pairs_set.add((ids[i], ids[j]))

    # Original blocking patterns (as supplement)
    print("Generating pairs from original name-based patterns...")
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
    print(f"Generated {len(candidate_pairs)} total candidate pairs from all strategies")

    # Sort by jaccard similarity using sempipes features (like baseline)
    print("Ranking pairs by similarity...")
    jaccard_similarities = []
    candidate_pairs_real_ids = []

    # Create id -> index mapping for fast lookup
    id_to_idx = {z2_with_features.loc[i, "id"]: i for i in range(len(z2_with_features))}

    for pair in candidate_pairs:
        id1, id2 = pair  # These are already real IDs from the blocking

        if id1 < id2:
            candidate_pairs_real_ids.append((id1, id2))
        else:
            candidate_pairs_real_ids.append((id2, id1))

        # Get indices for similarity computation
        idx1 = id_to_idx.get(id1)
        idx2 = id_to_idx.get(id2)

        if idx1 is None or idx2 is None:
            jaccard = 0.0
        else:
            # Compute jaccard similarity using ALL sempipes features
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

            # Fallback to original name if no features
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

    # Sort by similarity (highest first)
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]

    # Take top 2M pairs
    cand2 = candidate_pairs_real_ids[:2_000_000]
    print(f"Selected top {len(cand2)} pairs by similarity")

    return cand2


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
    print(f"Repeating blocking for Z2 {num_repeats} times...")

    all_cand2 = []
    Z2_recalls = []

    base_path = "experiments/sigmod/hidden_data"
    evaluation_dataset_path_Y2 = os.path.join(base_path, "Y2.csv")
    output_path = "experiments/sigmod/results/output_aide_sempipes.csv"

    for repeat_idx in range(num_repeats):
        print(f"\n--- Repeat {repeat_idx + 1}/{num_repeats} ---")

        # Perform blocking for Z2 with sempipes features (with different seed for each repeat)
        print(f"Performing blocking for Z2 with sempipes features (repeat {repeat_idx + 1})...")
        cand2 = compute_z2_blocking_with_features(seed=1 + repeat_idx, repeat_idx=repeat_idx)

        # Collect candidate pairs from this repeat
        all_cand2.extend(cand2)
        print(f"Found {len(cand2)} candidate pairs in repeat {repeat_idx + 1}")

        # Save output and evaluate Z2 recall for this iteration
        save_output(cand1, cand2, output_path)

        # Read from output and evaluate Z2
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
