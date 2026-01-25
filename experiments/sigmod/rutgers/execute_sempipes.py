import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import skrub
import unidecode
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
            "CRITICAL FOR RECALL: Brand is one of the most important features for blocking. Extract it whenever possible. "
            "You MUST extend this list with obvious aliases and typos (e.g. 'san disk' -> 'sandisk', 'san-disk' -> 'sandisk', "
            "'sandisc' -> 'sandisk', 'samsun' -> 'samsung', 'kingstn' -> 'kingston', 'toshbia' -> 'toshiba', 'transcent' -> 'transcend'). "
            "Handle multilingual variations: 'sandisk' in English, 'sandisk' in French/German/etc. (usually same spelling). "
            "Use EXACT regex: r'\\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\\b' "
            "BUT ALSO try fuzzy matching: if you see 'san disk', 'san-disk', 'sandisc', 'san-disc' -> normalize to 'sandisk'. "
            "Similarly: 'samsun' -> 'samsung', 'kingstn' -> 'kingston', 'toshbia' -> 'toshiba', 'transcent' -> 'transcend'. "
            "Use re.findall() to get ALL matches, deduplicate using set(), sort, join with single space. "
            "Examples: 'sandisk ultra 32gb' -> 'sandisk', 'kingston datatraveler' -> 'kingston', 'san disk usb' -> 'sandisk', "
            "'sandisc extreme' -> 'sandisk', 'samsun evo' -> 'samsung'. "
            "Return lowercase tokens only. Be EXTREMELY lenient - if something looks like a brand (even with typos or spacing), normalize and extract it. "
            "ONLY return '0' if you're absolutely certain there's no brand mentioned (very rare - most products have a brand)."
        ),
        "capacity": (
            "Extract the storage capacity from the name. CRITICAL FOR RECALL: Capacity is essential for blocking - same brand but different capacity = different product. "
            "Use regex over the normalized name to detect patterns like '32 gb', '64gb', '128 go', '1tb', '2 tb', '256gb', '512 gb', '1 tb', '128', '256', '512'. "
            "Handle multilingual units: 'gb' (English), 'go' (French), 'g' (abbreviation), 'gigaoctet' all mean gigabytes. Normalize all to 'gb'. "
            "Also handle: 'tb' (terabyte), 'to' (French terabyte), 't' (abbreviation) -> normalize to 'tb'. "
            "Use MULTIPLE regex patterns: "
            "1. r'([1-9]{1,3})[-\\s]*[g][bo]?' for GB patterns "
            "2. r'([1-9])[-\\s]*[t][bo]?' for TB patterns "
            "3. r'\\b([1-9]{1,3})\\s*(?:gb|go|g|tb|to|t)\\b' for explicit unit patterns "
            "4. r'\\b([1-9]{1,3})\\b' followed by checking if next word contains 'gb', 'go', 'g', 'tb', 'to', 't' "
            "For each match: "
            "1. Use re.findall() to get ALL matches from all patterns. "
            "2. For each match, apply re.sub('[^0-9a-z]+','', match) to remove non-alphanumeric, keeping digits and letters. "
            "3. Normalize 'go' -> 'gb', 'g' -> 'gb', 'to' -> 'tb', 't' -> 'tb' (if not already 'gb' or 'tb'). "
            "4. If no unit found but number is present, assume 'gb' for numbers < 10, 'tb' for numbers >= 10 (but prefer explicit units). "
            "5. Deduplicate using set(), sort, join with single space. "
            "Examples: 'sandisk ultra 32 gb usb' -> '32gb', '64gb 128gb bundle' -> '64gb 128gb', '256 go' -> '256gb', '128 g' -> '128gb', "
            "'1tb' -> '1tb', '2 to' -> '2tb', 'sandisk 128' -> '128gb' (if no explicit unit, assume gb for reasonable sizes). "
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
            "\n\nCRITICAL: Use unidecode.unidecode() to normalize ALL text to ASCII BEFORE any other processing. "
            "This is essential for handling multilingual text. "
            "\n\nLANGUAGE NORMALIZATION REQUIREMENTS:\n"
            "1. Capacity units: Normalize 'go', 'go', 'gigaoctet' → 'gb'; 'mo', 'mo' → 'mb'. "
            "   Handle all language variants of capacity units (French 'go', Spanish 'gb', etc.). "
            "2. Product type synonyms: Normalize 'carte mémoire'/'carte'/'tarjeta'/'karta'/'minneskort' → 'card'; "
            "   'clé USB'/'clé'/'pendrive'/'memoria USB' → 'usb' or 'stick'. "
            "3. Speed/Class: Normalize 'classe 10'/'clase 10'/'class 10'/'c10' → 'class10'; "
            "   'uhs-i'/'uhs-1'/'uhs i' → 'uhsi'. "
            "4. All text should be lowercase after normalization. "
            "\n\nEXTRACTION REQUIREMENTS:\n"
            "1. Follow the regex patterns and cleaning logic provided in each output column description EXACTLY. "
            "2. The patterns are from proven blocking code - use them VERBATIM, do NOT modify or simplify them. "
            "3. For each field, follow the EXACT extraction logic described (findall vs search, deduplication method, sorting, joining). "
            "4. Apply alias replacements EXACTLY as specified in the normalized_name description. "
            "5. Pre-compile ALL regex patterns using re.compile() for performance. "
            "6. Use the specified methods: re.findall() when description says 'find all', re.search() when it says 'first match'. "
            "7. For deduplication: use set() then convert back to sorted list. "
            "8. For joining: use single space ' '.join(). "
            "9. Return '0' (string zero) for missing values, NOT empty string or None. "
            "\n\nCRITICAL FOR RECALL - HIGHEST PRIORITY:\n"
            "Aim for MAXIMUM RECALL - it's MUCH better to extract something (even if slightly imperfect) than to return '0'. "
            "For blocking/entity matching, missing a feature (returning '0') causes missed matches and DRASTICALLY reduces recall. "
            "If a pattern is even REMOTELY close to matching, extract it. Only return '0' when you're ABSOLUTELY CERTAIN there's no match. "
            "Handle multilingual variations EXTREMELY aggressively - extract when in doubt, extract partial matches, extract variations. "
            "PREFER extracting multiple candidates over returning '0'. If unsure between two values, extract both (space-separated). "
            "For blocking, false positives are acceptable but false negatives (missing features) are catastrophic for recall. "
            "\n\nMODEL NUMBER EXTRACTION:\n"
            "Extract model numbers/SKUs aggressively using pattern: alphanumeric codes with hyphens/slashes "
            "(e.g., 'SDCZ50-064G-B35', 'MB-MG32DA/EU', 'SDSQUNC-032G-GN6IA'). "
            "These are often the strongest matching signals. "
            "Normalize by removing extra spaces and ensuring consistent format. "
            "Look for patterns like: [A-Z]{2,}[0-9A-Z-]+ or similar alphanumeric codes. "
            "\n\nTEXT CLEANING:\n"
            "Remove URLs, HTML tags, seller information (e.g., 'amazon.es:', 'fnac.es', 'tesco direct:') more aggressively. "
            "Clean up whitespace and special characters. "
            "\n\nPERFORMANCE: Optimize for throughput - must handle millions of rows within 15 minutes. "
            "Pre-compile regex patterns, avoid repeated compilation, use efficient string operations. "
            "\n\nIn comments, explain your code and why you chose the approach you did."
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
                "You are analyzing MULTILINGUAL product titles of storage devices (USB sticks, SD cards, SSDs, etc.) for entity blocking/matching. "
                "The goal is to find duplicate products that may be described differently in different languages or with different formatting. "
                f"\n\nEXISTING COLUMNS ALREADY EXTRACTED: {existing_cols_str}. "
                "These are used for blocking patterns like: brand+capacity, brand+capacity+type, brand+model, brand+features, model_long, model_short. "
                "\n\nTASK: Suggest 1-2 ADDITIONAL features that would help match duplicate products across languages. "
                "Focus on features that are COMMONLY present (>10% of rows), language-independent, and can be extracted using regex/rule-based methods. "
                "\n\nHIGH PRIORITY CANDIDATES:\n"
                "1. Speed/Class ratings: Extract and normalize 'Class 10'/'Classe 10'/'clase 10'/'C10' to 'class10'; "
                "   'UHS-I'/'UHS-1'/'uhs-i'/'UHS I' to 'uhsi'; speed in MB/s (normalize units: 'mb/s', 'mbps', 'mo/s' to 'mbs'). "
                "   Use unidecode and handle all language variants. "
                "2. Model numbers/SKUs: Extract alphanumeric codes with hyphens (e.g., 'SDCZ50-064G-B35', 'MB-MG32DA/EU', 'SDSQUNC-032G-GN6IA'). "
                "   Look for patterns like uppercase letters followed by numbers and hyphens. These are very strong matching signals. "
                "3. Form factors: Normalize 'microSD'/'micro SD'/'micro-sd' to 'microsd'; 'SDXC'/'SD XC' to 'sdxc'. "
                "   Extract consistently: 'microsd', 'microsdhc', 'microsdxc', 'sd', 'sdhc', 'sdxc', 'usb', 'xqd', etc. "
                "\n\nMEDIUM PRIORITY:\n"
                "4. Adapter information: 'with adapter'/'con adaptador'/'mit adapter' to boolean or normalized text. "
                "5. Package type: 'FFP'/'Frustration-Free Packaging' normalization. "
                "\n\nCRITICAL JSON FORMATTING REQUIREMENTS:\n"
                "When providing regex patterns in the feature_prompt field, you MUST properly escape all backslashes for JSON. "
                "In JSON strings, a single backslash must be written as two backslashes. "
                "For example: use '\\\\d' (four characters: backslash-backslash-d) instead of '\\d' (two characters: backslash-d). "
                "Similarly, '\\\\w' instead of '\\w', '\\\\s' instead of '\\s', etc. "
                "Test your JSON before returning it - it must be valid JSON that can be parsed by json.loads(). "
                "\n\nFor each feature, provide: feature_name (lowercase, underscore-separated), "
                "feature_prompt (detailed extraction instructions with regex patterns, ensuring all backslashes are doubled for JSON, and language normalization using unidecode), "
                "and input_columns (list of columns to use). "
                "Return valid JSON array format as specified in the system prompt."
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
            jaccard_similarities.append(len(s1.intersection(s2)) / min(len(s1), len(s2)))

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
    output_df.to_csv("output.csv", index=False)


def save_output_X1_from_file(X1_candidate_pairs_df, X2_candidate_pairs):
    expected_cand_size_X2 = 2000000

    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    X2_candidate_pairs_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])

    output_df = pd.concat([X1_candidate_pairs_df, X2_candidate_pairs_df], ignore_index=True)
    output_df.to_csv("output.csv", index=False)


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
    output_path = "output.csv"
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
