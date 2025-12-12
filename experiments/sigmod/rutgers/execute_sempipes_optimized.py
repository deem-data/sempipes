import os
import re

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator
from skrub._data_ops._evaluation import find_node_by_name

import sempipes
from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label
from experiments.sigmod.rutgers.execute_sempipes import block_with_attr, save_output_X1_from_file
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
            "Clean and normalize the product name following these EXACT steps in order: "
            "1. Convert input to string: str(name) "
            "2. Apply unidecode.unidecode() to convert to ASCII (handles accented characters like é, ü, etc.) "
            "3. Convert to lowercase: .lower() "
            "4. Apply alias replacements using dictionary lookup (iterate through x2_aliases dict): "
            "   - For each key-value pair, replace each value in the list with the key "
            "   - Aliases: 'classe/clase/clas /klasse/cl ' -> 'class', 'uhs1/uhs-i/ultra high-speed' -> 'uhsi', "
            "     'typec/type c/usb-c/usbc' -> 'type-c', 'hyperx/savage' -> 'kingston hxs', "
            "     'serie ux' -> 'sony g1ux', 'dtig4/ 101 /dt101g2' -> ' kingston dt101 ', "
            "     'sda10/sda3' -> ' kingston ultimate ', 'extrem ' -> 'extreme ', 'attach' -> 'att4' "
            "5. Compile and apply removal regex: r'&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s][m]*(?=[23][\.\s])|(?<=usb)-[\w]+\s(?=[23][\.\s])|(?<=[a-z])[\s]+gb|(?<=data|jump)[t\s](?=trave|drive)|(?<=extreme|exceria)[\s](?=pro[\s]|plus)|(?<=class)[\s_](?=10|[234]\b)|(?<=gen)[\s_](?=[134\s][0]*)' using re.sub() "
            "6. Compile and apply class10 replacement regex: r'(10 class|class 10|class(?=[\w]+10\b)|cl\s10)' -> replace with 'class10' "
            "7. String replacements: name.replace('class 4 ', 'class4 '), name.replace('class 3 ', 'class3 ') "
            "8. Replace multiple spaces: name.replace('  ', ' ') (repeat until no more double spaces) "
            "Return the final cleaned string. This normalized name is used for all subsequent feature extraction."
        ),
        "brand": (
            f"Extract the main storage brand from the name. The minimal set of brands is {brands_list}. "
            "CRITICAL FOR RECALL - HIGHEST PRIORITY: Brand is THE SINGLE MOST IMPORTANT feature for blocking. "
            "Missing a brand causes catastrophic recall loss. Extract it whenever possible, even with severe typos or spacing issues. "
            "You MUST extend this list with obvious aliases and typos (e.g. 'san disk' -> 'sandisk', 'san-disk' -> 'sandisk', "
            "'sandisc' -> 'sandisk', 'samsun' -> 'samsung', 'kingstn' -> 'kingston', 'toshbia' -> 'toshiba', 'transcent' -> 'transcend'). "
            "Handle multilingual variations: 'sandisk' in English, 'sandisk' in French/German/etc. (usually same spelling). "
            "Use EXACT regex: r'\\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\\b' "
            "BUT ALSO try fuzzy matching: if you see 'san disk', 'san-disk', 'sandisc', 'san-disc' -> normalize to 'sandisk'. "
            "Similarly: 'samsun' -> 'samsung', 'kingstn' -> 'kingston', 'toshbia' -> 'toshiba', 'transcent' -> 'transcend'. "
            "ALSO handle partial matches: if you see 'sandis' -> likely 'sandisk', 'samsu' -> likely 'samsung'. "
            "Use re.findall() to get ALL matches, deduplicate using set(), sort, join with single space. "
            "Examples: 'sandisk ultra 32gb' -> 'sandisk', 'kingston datatraveler' -> 'kingston', 'san disk usb' -> 'sandisk', "
            "'sandisc extreme' -> 'sandisk', 'samsun evo' -> 'samsung', 'san disk extreme pro' -> 'sandisk'. "
            "Return lowercase tokens only. Be EXTREMELY lenient - if something looks like a brand (even with typos, spacing, or partial match), normalize and extract it. "
            "ONLY return '0' if you're absolutely certain there's no brand mentioned (very rare - most products have a brand)."
        ),
        "capacity": (
            "Extract the storage capacity from the name. CRITICAL FOR RECALL: Capacity is essential for blocking - same brand but different capacity = different product. "
            "Missing capacity causes significant recall loss. "
            "Use regex over the normalized name to detect patterns like '32 gb', '64gb', '128 go', '1tb', '2 tb', '256gb', '512 gb', '1 tb', '128', '256', '512'. "
            "Handle multilingual units: 'gb' (English), 'go' (French), 'g' (abbreviation), 'gigaoctet' all mean gigabytes. Normalize all to 'gb'. "
            "Also handle: 'tb' (terabyte), 'to' (French terabyte), 't' (abbreviation) -> normalize to 'tb'. "
            "Use MULTIPLE regex patterns: "
            "1. r'([1-9]{1,3})[-\\s]*[g][bo]?' for GB patterns "
            "2. r'([1-9])[-\\s]*[t][bo]?' for TB patterns "
            "3. r'\\b([1-9]{1,3})\\s*(?:gb|go|g|tb|to|t)\\b' for explicit unit patterns "
            "4. r'\\b([1-9]{1,3})\\b' followed by checking if next word contains 'gb', 'go', 'g', 'tb', 'to', 't' "
            "5. For standalone numbers without units: Look for common capacity sizes (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) and assume 'gb' "
            "For each match: "
            "1. Use re.findall() to get ALL matches from all patterns. "
            "2. For each match, apply re.sub('[^0-9a-z]+','', match) to remove non-alphanumeric, keeping digits and letters. "
            "3. Normalize 'go' -> 'gb', 'g' -> 'gb', 'to' -> 'tb', 't' -> 'tb' (if not already 'gb' or 'tb'). "
            "4. If no unit found but number is present, assume 'gb' for numbers < 10, 'tb' for numbers >= 10 (but prefer explicit units). "
            "5. For standalone numbers in common capacity sizes (1, 2, 4, 8, 16, 32, 64, 128, 256, 512), assume 'gb' if no unit found. "
            "6. Deduplicate using set(), sort, join with single space. "
            "Examples: 'sandisk ultra 32 gb usb' -> '32gb', '64gb 128gb bundle' -> '64gb 128gb', '256 go' -> '256gb', '128 g' -> '128gb', "
            "'1tb' -> '1tb', '2 to' -> '2tb', 'sandisk 128' -> '128gb' (if no explicit unit, assume gb for reasonable sizes), "
            "'kingston 64' -> '64gb' (standalone number in common size). "
            "If multiple capacities appear, return all (space-separated). "
            "Be EXTREMELY lenient - extract ANY number that could be a capacity, even if format is unusual. "
            "ONLY return '0' if you're absolutely certain there's no capacity mentioned (very rare - most products list capacity)."
        ),
        "mem_type": (
            "Extract product type using EXACT logic (NOT just regex): "
            "1. First, search for regex r'xqd|ssd|tv|lte' in the normalized name. "
            "2. If 'lte' is found, return 'phone' (not 'lte'). "
            "3. If no match from step 1, check if 'fdrive' is in the text (case-insensitive) - if yes, return 'fdrive'. "
            "4. If still no match, check if 'memcard' is in the text (case-insensitive) - if yes, return 'memcard'. "
            "5. If a match from step 1 exists and it's not 'lte', return that match (e.g., 'xqd', 'ssd', 'tv'). "
            "6. Otherwise return '0'. "
            "Return single lowercase token only."
        ),
        "type": (
            f"Extract a short product type / line relative to the brand, based on the family keywords mapping {families_dict}, "
            f"the Intenso-specific list {intenso_type}, and Samsung color names {colors}. "
            "Use EXACT regex: r'\\b(datatraveler|extreme[p]?|exceria[p]?|dual[\\s]*(?!=sim)|evo|xqd|ssd|cruzer[\\w+]*|glide|blade|basic|fit|force|basic line|jump\\s?drive|hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|beast|fury|impact|a400|sd[hx]c|uhs[i12][i1]*|note\\s?9|ultra|premium|basic|flash|plus|otg|xpro|xhigh|midnight black|prism white|prism black|prism green|prism blue)\\b'. "
            "Use re.findall() to get ALL matches, deduplicate using set(), sort, join with single space. "
            "Examples: 'sandisk extreme pro 32gb' -> 'extreme', 'kingston datatraveler' -> 'datatraveler', "
            "'toshiba exceria plus' -> 'exceria', 'lexar jumpdrive' -> 'jumpdrive', 'sandisk cruzer fit' -> 'cruzer', "
            "'samsung galaxy midnight black' -> 'midnight black', 'intenso premium line' -> 'premium'. "
            "Be lenient - if you see 'extreme pro', extract 'extreme'. If you see 'cruzer fit', extract 'cruzer'. "
            "Return '0' only if you're very confident there's no type/line information."
        ),
        "model": (
            "Extract a concise model identifier. Use regexes to catch hyphenated or letter+digit codes. "
            "Use EXACT regex: r'\\b([\\(]*[\\w]+[-]*[\\d]+[-]*[\\w]+[-]*[\\d+]*|[\\d]+[\\w]|[\\w][\\d]+)\\b'. "
            "Use re.findall() to get ALL matches, filter to len(match) >= 3 (RELAXED from 5 for better recall), "
            "deduplicate using set(), sort, join with space. "
            "Examples: 'sandisk extreme pro sdxc' -> 'extreme pro', 'kingston dt101g2' -> 'dt101g2', "
            "'toshiba u202' -> 'u202' (now included with len >= 3), 'samsung evo plus' -> 'evo plus', "
            "'dt101' -> 'dt101', 'sda10' -> 'sda10'. "
            "Be EXTREMELY lenient - if you see model-like patterns (letters+numbers, hyphenated codes, short codes), extract them. "
            "Even short codes like 'u202', 'dt101', 'sda10' are valuable for blocking. "
            "ONLY return '0' if you're very confident there's no model identifier (rare - most products have some identifier)."
        ),
        "model_long": (
            "Extract LONG model number patterns using EXACT regex: r'(thn-[a-z][\\w]+|ljd[\\w+][-][\\w]+|ljd[sc][\\w]+[-][\\w]+|lsdmi[\\d]+[\\w]+|lsd[0-9]{1,3}[gb]+[\\w]+|ljds[0-9]{2}[-][\\w]+|usm[0-9]{1,3}[\\w]+|sdsq[a-z]+[-][0-9]+[a-z]+[-][\\w]+|sdsd[a-z]+[-][0-9]+[\\w]+[-]*[\\w]*|sdcz[\\w]+|mk[\\d]+|sr-g1[\\w]+)'. "
            "ALSO use a more general pattern for any alphanumeric code with hyphens/slashes: "
            "r'([A-Z]{2,}[0-9A-Z-]+(?:[/-][0-9A-Z-]+)*)' for uppercase model codes like 'SDCZ50-064G-B35', 'MB-MG32DA/EU'. "
            "Use re.findall() to get ALL matches (not just first), deduplicate, sort, join with space. "
            "Examples: 'sdsqxa-128g-anc' -> 'sdsqxa-128g-anc', 'usm32gqx' -> 'usm32gqx', 'mk123456' -> 'mk123456', "
            "'SDCZ50-064G-B35' -> 'SDCZ50-064G-B35', 'MB-MG32DA/EU' -> 'MB-MG32DA/EU'. "
            "Be EXTREMELY lenient - if you see ANY long alphanumeric pattern with hyphens/slashes that looks like a model number, extract it. "
            "Return '0' only if absolutely no such patterns found."
        ),
        "model_short": (
            "Extract SHORT model number patterns using EXACT regex: r'\\b(c20[mc]|sda[0-9]{1,2}|g1ux|s[72][05]|[unm][23]02|p20|g4|dt101|se9|[asm][0-9]{2})\\b'. "
            "Use re.findall() to find ALL matches, deduplicate using set(), sort lexicographically, join with space. "
            "Examples: 'sda10' -> 'sda10', 'dt101' -> 'dt101', 'g1ux' -> 'g1ux', 'c20m' -> 'c20m'. "
            "Return '0' if none found. Be lenient - extract short codes that match the pattern."
        ),
        "features": (
            "Extract technical features using EXACT regex: r'\\b(usb[23]|type-c|uhs[i]{1,2}|class[0134]{1,2}|gen[1-9]{1,2}|u[23](?=[\\s\\.])|sd[hx]c|otg|lte|[45]g[-\\s]lte|[0-9]+(?=-inch)|[0-9]{2,3}r[0-9]{2,3}w|[0-9]{2,3}(?=[\\smbo/p]{3}))\\b'. "
            "CRITICAL FOR RECALL: Features help distinguish products. Extract all technical specs you find. "
            "Handle multilingual: 'usb 3.0' (English), 'usb 3.0' (same in most languages), 'classe 10' -> 'class10' (after normalization). "
            "Use re.findall() to get ALL matches, deduplicate using set(), sort, join with single space. "
            "This matches the original x2_feature_reg pattern exactly. "
            "Examples: 'usb 3.0 type-c' -> 'type-c usb3', 'uhs-i class10' -> 'class10 uhs-i', "
            "'sdhc card 95r80w' -> 'sdhc 95r80w', 'usb 2.0' -> 'usb2'. "
            "Be lenient - if you see 'usb3', 'usb 3', 'usb-3.0', extract as 'usb3'. "
            "Return '0' only if absolutely no technical features found."
        ),
        "item_code": (
            "Extract explicit numeric item codes, especially those in parentheses, using regex like r'\\((mk)?[0-9]{6,10}\\)'. "
            "Strip parentheses and any 'mk' prefix, leaving only the digits. "
            "Also look for patterns like '(4187407)', '(mk483394661)', '(173473)', etc. "
            "If multiple codes appear, pick the one most likely to be a manufacturer part number (usually longer, in parentheses). "
            "Return the digits as a string, or '0' if none found."
        ),
        "series": (
            f"Extract the product series / family token. Use the brand-specific families mapping {families_dict} "
            f"and the Intenso-specific list {intenso_type} and Samsung color names {colors} over the normalized name. "
            "Normalize obvious typos ('cruizer' -> 'cruzer'). "
            "Return a single lowercase family token (e.g. 'glide', 'cruzer', 'ultimate', 'exceria', 'jumpdrive', 'premium', 'basic'), "
            "or '0' if none."
        ),
        "pat_hb": (
            "Extract the first hyphenated alphanumeric pattern, using regex r'\\w+-\\w+' over the normalized name. "
            "Typical examples are 'uhs-i', 'class-10', 'high-speed', 'type-c', 'usb-3', 'micro-sd'. "
            "Return the first such pattern in lowercase, or '0' if none."
        ),
        "hybrid": (
            "Extract all 'hybrid' identifiers that mix letters and digits and are at least length 5, similar to the "
            "regex r'(?=[^\\W\\d_]*\\d)(?=\\d*[^\\W\\d_])[^\\W_gGM]{5,}'. "
            "This matches alphanumeric codes like 'dt101g2', 'sda10', 'usm32gqx', 'lsd16gcrbeu1000', etc. "
            "Use regex on the normalized name, deduplicate matches, sort them lexicographically, and join with a single space. "
            "Return '0' if there are no such identifiers."
        ),
        "long_num": (
            "Extract all sequences of 4 or more digits using regex r'[0-9]{4,}' on the normalized name. "
            "Deduplicate the numbers, sort them lexicographically, and join with a single space. "
            "Examples: '4187407' -> '4187407', '483394661' -> '483394661', '173473' -> '173473'. "
            "Return '0' if none are found."
        ),
    }

    data_ref = data_ref.sem_extract_features(
        nl_prompt=(
            "YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. Use ONLY regex and rule-based approaches, "
            "DO NOT USE ANY Transformer, NER, LLM fallbacks, or LM models. "
            "You are given very dirty MULTILINGUAL titles of storage products (USB sticks, SD / microSD cards, SSDs, SIM cards, Samsung phones / TVs, etc.). "
            "The data contains product names in multiple languages (English, French, German, Spanish, Italian, Polish, etc.). "
            "Product names may contain accented characters, special characters, and language-specific formatting. "
            "\n\nCRITICAL: Normalize ALL text to ASCII BEFORE any other processing. "
            "This is essential for handling multilingual text. "
            "IMPORTANT: Use `unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')` "
            "instead of unidecode (unidecode may not be available in the execution environment). "
            "This handles accented characters like é→e, ü→u, etc. "
            "\n\nLANGUAGE NORMALIZATION REQUIREMENTS:\n"
            "1. Capacity units: Normalize 'go', 'go', 'gigaoctet' → 'gb'; 'mo', 'mo' → 'mb'. "
            "   Handle all language variants of capacity units (French 'go', Spanish 'gb', etc.). "
            "2. Product type synonyms: Normalize 'carte mémoire'/'carte'/'tarjeta'/'karta'/'minneskort' → 'card'; "
            "   'clé USB'/'clé'/'pendrive'/'memoria USB' → 'usb' or 'stick'. "
            "3. Speed/Class: Normalize 'classe 10'/'clase 10'/'class 10'/'c10' → 'class10'; "
            "   'uhs-i'/'uhs-1'/'uhs i' → 'uhsi'. "
            "4. All text should be lowercase after normalization. "
            "\n\nCODE STRUCTURE PATTERNS (FROM SUCCESSFUL RUNS):\n"
            "The most successful implementations (achieving 0.257-0.261 recall) used these patterns: "
            "1. EITHER class-based OR function-based approach works: "
            "   - Class: Create a class (e.g., 'FeatureExtractor') with methods (_normalize_name, _extract_brand, etc.) "
            "   - Functions: Use standalone functions (_normalize_text, _extract_brand, etc.) at module level "
            "2. Pre-compile ALL regex patterns at module/class level using re.compile() - this is critical for performance "
            "   Example: RE_BRAND = re.compile(r'\\b(intenso|lexar|...)\\b') at module/class level, NOT inside methods "
            "3. Initialize all output columns to '0' (string) BEFORE extraction to ensure no missing values "
            "   Example: for col in feature_cols: df[col] = '0' "
            "4. Use .progress_apply() from tqdm for applying extraction functions (enables progress bars) "
            "   Example: tqdm.pandas(desc='Extracting features'); df['brand'] = df['normalized_name'].progress_apply(_extract_brand) "
            "5. Extract normalized_name FIRST, then use it for all subsequent extractions "
            "   Example: df['normalized_name'] = df['name'].progress_apply(_normalize_text) "
            "   Then: df['brand'] = df['normalized_name'].progress_apply(_extract_brand) "
            "6. Use helper methods/functions for each extraction type - keeps code organized and maintainable "
            "7. Each extraction function should take normalized_name as input and return a string (either extracted value or '0') "
            "8. Use compiled regex objects throughout - access them as module/class variables, never recompile "
            "9. Sort alias replacements by length descending - ensures longer, more specific aliases are replaced first "
            "   Example: alias_replacements_flat.sort(key=lambda x: len(x[0].pattern), reverse=True) "
            "10. Create helper function for processing matches: deduplicate using set(), sort, join with space "
            "    Example: def _process_matches(matches): return ' '.join(sorted(set(matches))) if matches else '0' "
            "11. For capacity extraction, use tokenization approach: split text into tokens, check if number is followed by unit "
            "    Example: tokens = normalized_name.split(); check tokens[i+1] for unit if tokens[i] is a number "
            "12. Handle edge cases: Check if normalized_name == '0' at start of each extraction function and return '0' early "
            "13. Use print statements for progress feedback: print('Extracting brand...'), print('Extracting capacity...'), etc. "
            "\n\nEXTRACTION REQUIREMENTS:\n"
            "1. Follow the regex patterns and cleaning logic provided in each output column description EXACTLY. "
            "2. The patterns are from proven blocking code - use them VERBATIM, do NOT modify or simplify them. "
            "3. For each field, follow the EXACT extraction logic described (findall vs search, deduplication method, sorting, joining). "
            "4. Apply alias replacements EXACTLY as specified in the normalized_name description. "
            "5. Pre-compile ALL regex patterns using re.compile() at module/class level (NOT inside loops) for performance. "
            "6. Use the specified methods: re.findall() when description says 'find all', re.search() when it says 'first match'. "
            "7. For deduplication: use set() then convert back to sorted list. "
            "8. For joining: use single space ' '.join(). "
            "9. Return '0' (string zero) for missing values, NOT empty string or None. "
            "10. Initialize all columns to '0' BEFORE extraction, then overwrite with extracted values. "
            "\n\nCRITICAL FOR RECALL - HIGHEST PRIORITY:\n"
            "Aim for MAXIMUM RECALL - it's MUCH better to extract something (even if slightly imperfect) than to return '0'. "
            "For blocking/entity matching, missing a feature (returning '0') causes missed matches and DRASTICALLY reduces recall. "
            "If a pattern is even REMOTELY close to matching, extract it. Only return '0' when you're ABSOLUTELY CERTAIN there's no match. "
            "Handle multilingual variations EXTREMELY aggressively - extract when in doubt, extract partial matches, extract variations. "
            "PREFER extracting multiple candidates over returning '0'. If unsure between two values, extract both (space-separated). "
            "For blocking, false positives are acceptable but false negatives (missing features) are catastrophic for recall. "
            "\n\nSPECIFIC RECALL IMPROVEMENTS BASED ON BASELINE ANALYSIS:\n"
            "1. BRAND EXTRACTION: This is THE MOST CRITICAL feature for blocking. Extract brands even with typos, spacing issues, or partial matches. "
            "   Use a TWO-STEP approach for maximum recall: "
            "   a) First, process fuzzy aliases (e.g., 'san disk' → 'sandisk', 'san-disk' → 'sandisk', 'sandisc' → 'sandisk', 'samsun' → 'samsung') "
            "   b) Then, apply main brand regex on the normalized text "
            "   c) Sort aliases by length descending to match longer patterns first "
            "   d) Use word boundaries (\\b) in regex patterns to ensure whole-word matches "
            "   Examples: 'san disk' → 'sandisk', 'san-disk' → 'sandisk', 'sandisc' → 'sandisk', 'samsun' → 'samsung'. "
            "   If you see ANYTHING that looks like a brand name, extract it. "
            "2. CAPACITY EXTRACTION: Extract ALL capacity mentions, even if format is unusual. Use MULTIPLE strategies: "
            "   a) Explicit unit patterns: '32 gb', '64gb', '128 go', '1tb' "
            "   b) Tokenization approach: Split text into tokens, check if number is followed by unit in next token "
            "   c) Standalone numbers: For common capacity sizes (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 1024), assume 'gb' if no unit found "
            "   d) Handle cases like '128', '256', '512' without explicit units - assume 'gb' for reasonable sizes "
            "   e) Extract multiple capacities if present (e.g., '64gb 128gb bundle' → extract both) "
            "   f) Normalize units: 'go'/'g' → 'gb', 'to'/'t' → 'tb' "
            "3. MODEL EXTRACTION: Extract model numbers VERY aggressively. Short codes (3+ chars) are valuable: 'u202', 'dt101', 'sda10', 'g1ux'. "
            "   Long hyphenated patterns are critical: 'SDCZ50-064G-B35', 'MB-MG32DA/EU', 'SDSQUNC-032G-GN6IA'. "
            "   Also extract alphanumeric patterns: 'dt101g2', 'usm32gqx', 'lsd16gcrbeu1000'. "
            "4. FEATURES EXTRACTION: Extract ALL technical features: USB versions, UHS classes, speed ratings, form factors. "
            "   These help distinguish similar products. Be lenient: 'usb3', 'usb 3', 'usb-3.0' all → 'usb3'. "
            "   Use the main features regex, then add aggressive matching for common variations: "
            "   - Check for 'usb 3.0', 'usb 3', 'usb-3.0' patterns and normalize to 'usb3' "
            "   - Check for 'usb 2.0', 'usb 2', 'usb-2.0' patterns and normalize to 'usb2' "
            "   - The normalized_name already has 'uhs-i' → 'uhsi' and 'class 10' → 'class10' from alias replacements "
            "5. SERIES/TYPE EXTRACTION: Extract product lines/families aggressively: 'extreme', 'cruzer', 'datatraveler', 'evo', 'exceria'. "
            "   These are strong matching signals. Handle variations: 'extreme pro' → extract 'extreme', 'cruzer fit' → extract 'cruzer'. "
            "\n\nMULTILINGUAL HANDLING - CRITICAL FOR RECALL:\n"
            "The dataset contains products in multiple languages. You MUST handle: "
            "- French: 'go' (gigaoctet), 'clé USB', 'carte mémoire', 'classe 10' "
            "- German: 'speicherkarte', 'usb-stick', 'klasse 10' "
            "- Spanish: 'tarjeta', 'memoria USB', 'clase 10' "
            "- Italian: 'carta', 'chiavetta USB', 'classe 10' "
            "- Polish and other languages: similar patterns "
            "Normalize ALL of these to English equivalents BEFORE extraction. Use unidecode FIRST, then apply language-specific normalizations. "
            "\n\nHYPHENATED PATTERN EXTRACTION (CRITICAL FOR RECALL):\n"
            "The blocking code uses hyphenated alphanumeric patterns extensively. Extract ALL patterns matching r'\\w+-\\w+' from normalized_name. "
            "Examples: 'uhs-i', 'class-10', 'high-speed', 'type-c', 'usb-3', 'micro-sd', 'sdcx50-064g-b35'. "
            "These patterns are stored in 'pat_hb' and are used directly in blocking. Extract them ALL, not just the first one. "
            "\n\nTEXT CLEANING:\n"
            "Remove URLs, HTML tags, seller information (e.g., 'amazon.es:', 'fnac.es', 'tesco direct:') more aggressively. "
            "Clean up whitespace and special characters. "
            "BUT: Do NOT remove product identifiers, model numbers, or technical specifications during cleaning. "
            "\n\nPERFORMANCE OPTIMIZATION (CRITICAL):\n"
            "Must handle millions of rows within 15 minutes. Key optimizations: "
            "1. Pre-compile ALL regex patterns at module/class level (NOT inside loops or functions) - this is the #1 performance factor "
            "2. Use compiled regex objects (from re.compile()) throughout, never compile inside loops "
            "3. Use .progress_apply() from tqdm for vectorized-like operations with progress tracking "
            "4. Avoid repeated string operations - normalize once, reuse normalized_name "
            "5. Use efficient pandas operations - avoid iterating row-by-row if possible "
            "6. Store compiled regex patterns as class/module variables, not recreated each call "
            "\n\nVALIDATION CHECKLIST:\n"
            "Before returning results, verify: "
            "1. Brand is extracted for most products (most products have a brand) "
            "2. Capacity is extracted for most products (most products list capacity) "
            "3. At least one model identifier (model, model_long, model_short, hybrid, or long_num) is extracted for most products "
            "4. normalized_name is never empty (always has some cleaned text) "
            "If extraction seems too conservative, be more aggressive. "
            "\n\nIn comments, explain your code and why you chose the approach you did."
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
            "You are analyzing MULTILINGUAL product titles of storage devices (USB sticks, SD cards, SSDs, etc.) for entity blocking/matching. "
            "The goal is to find duplicate products that may be described differently in different languages or with different formatting. "
            "\n\nEXISTING COLUMNS ALREADY EXTRACTED: brand, capacity, normalized_name, mem_type, type, model, model_long, model_short, "
            "features, item_code, series, pat_hb, hybrid, long_num. "
            "These are used for blocking patterns like: brand+capacity, brand+capacity+type, brand+model, brand+features, "
            "model_long, model_short, hybrid, long_num, series, item_code. "
            "\n\nTASK: Generate 3-5 ADDITIONAL features that would help match duplicate products across languages. "
            "Focus on features that are COMMONLY present (>10% of rows), language-independent, and can be extracted using regex/rule-based methods. "
            "\n\nHIGH PRIORITY CANDIDATES:\n"
            "1. Speed/Class ratings: Extract and normalize 'Class 10'/'Classe 10'/'clase 10'/'C10' to 'class10'; "
            "   'UHS-I'/'UHS-1'/'uhs-i'/'UHS I' to 'uhsi'; speed in MB/s (normalize units: 'mb/s', 'mbps', 'mo/s' to 'mbs'). "
            "2. Form factors: Normalize 'microSD'/'micro SD'/'micro-sd' to 'microsd'; 'SDXC'/'SD XC' to 'sdxc'. "
            "   Extract consistently: 'microsd', 'microsdhc', 'microsdxc', 'sd', 'sdhc', 'sdxc', 'usb', 'xqd', etc. "
            "3. Package/bundle information: Extract 'bundle', 'pack', 'pack of', 'lot of', etc. "
            "4. Adapter information: 'with adapter'/'con adaptador'/'mit adapter' to boolean or normalized text. "
            "5. Color information: Extract color names that might help distinguish products. "
            "\n\nGenerate Python code that extracts these features using regex and rule-based methods. "
            "Return features as lowercase strings, '0' for missing values. "
            "Optimize for throughput - must handle millions of rows efficiently."
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
        operator_name,
        scoring=recall_scorer_with_labels,
        cv=5,
        num_trials=24,
        search=EvolutionarySearch(population_size=6),
        additional_env_variables={
            "data_original_x2": train_X,
            "dummy_y": pd.Series([0] * len(train_X)),
            f"sempipes_prefitted_state__{operator_name2}": fixed_extract_state,  # Use pre-fitted state
        },
    )

    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
    print(f"Best outcome score after optimization on train CV: {best_outcome.score}, state: {best_outcome.state}")

    # Use optimized state for final prediction
    # Create environment with both the optimized state for sem_gen_features and the fixed state for sem_extract_features
    def _create_env_with_fixed_extract(X, y, operator_name, operator_name2, gen_features_state, extract_features_state):
        env = _create_env(X, y, operator_name, operator_name2, gen_features_state)
        env[f"sempipes_prefitted_state__{operator_name2}"] = extract_features_state
        return env

    learner_optimized = pipeline_to_optimise.skb.make_learner(fitted=False, keep_subsampling=False)
    learner_optimized.fit(
        _create_env_with_fixed_extract(
            train_X, train_labels, operator_name, operator_name2, best_outcome.state, fixed_extract_state
        )
    )
    optimized_results = learner_optimized.predict(
        _create_env_with_fixed_extract(
            test_data, test_labels, operator_name, operator_name2, best_outcome.state, fixed_extract_state
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
