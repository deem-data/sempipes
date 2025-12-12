import re

import pandas as pd
import skrub

import sempipes


def extract_x2_sempipes(data: pd.DataFrame, name: str = "extract_x2_features") -> pd.DataFrame | skrub.DataOp:
    """
    Clean `X2.csv` data into a feature table using `sempipes`, similar to `clean_sempipes_naive`.

    The output schema is the one expected by `EntityBlocking.block_x2`:

    :return:
        A DataFrame which contains following columns:
        {instance_id: instance_id of items;
         brand: item's brand, for example: {'intenso', 'pny', 'lexar'}
         capacity: usb/sd card's capacity, unit in GB
         price: price of the item
         mem_type: memory type, for example: {'ssd', 'sd', 'microsd', 'usb'}
         type: type information, relative to brand
         model: model information, relative to brand
         item_code: the unique item code
         title: title information of instance}
    """
    # Configure sempipes – keep it symmetric with the X1 pipeline.
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 1.5},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 1.5},
        ),
    )

    # Keep full original dataframe (id, name, price, …) available in the graph.
    data_ref = skrub.var("data_original_x2", data)

    # First: normalize the very dirty `name` field to make regexes / pattern extraction easier.
    data_ref = data_ref.sem_extract_features(
        nl_prompt=(
            "YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. Use ONLY regex and rule-based approaches, "
            "DO NOT USE ANY Transformer, NER, LLM fallbacks, or LM models. "
            "Cleaned and normalized version of the product name, suitable for downstream regex-based feature extraction and blocking. "
            "Use simple text cleaning: remove extra whitespace, normalize to lowercase, remove special characters if needed, "
            "but preserve alphanumeric content and structure. Use regex-based string operations only."
        ),
        input_columns=["name"],
        name="extract_normalized_name",
        output_columns={
            "normalized_name": "Cleaned and normalized version of the product name, suitable for downstream regex-based feature extraction and blocking."
        },
        generate_via_code=True,
    )

    # Brand / family / type helpers used in the original hand-written implementation.
    brands = ["sandisk", "lexar", "kingston", "intenso", "toshiba", "sony", "pny", "samsung", "transcend"]
    families = {
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
            "jumpdrive",
            "usb",
            "memo",
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
            "exceria",
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
            "micro",
            "line",
            "scheda",
            "usb",
            "sd",
            "premium",
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
        "0": [
            "adapt",
            "alu",
            "attach",
            "blade",
            "canvas",
            "cart",
            "cruzer",
            "cs/ultra",
            "datatravel",
            "evo",
            "exceria",
            "extern",
            "extreme",
            "flair",
            "flash",
            "galaxy",
            "glide",
            "hyperx",
            "jumpdrive",
            "kart",
            "klasse",
            "line",
            "memo",
            "memoria",
            "multi",
            "origin",
            "pendrive",
            "premium",
            "react",
            "scheda",
            "secure",
            "select",
            "serie",
            "speicher",
            "tarjeta",
            "transmemo",
            "transmemory",
            "traveler",
            "ultimate",
            "ultra",
            "usb",
            "usm32gqx",
            "veloc",
            "wex",
            "xqd",
        ],
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

    # Second: extract blocking features using deterministic regex-heavy code.
    # These columns are fed into `EntityBlocking.block_x2` and used in equality
    # comparisons, so keep them short, canonical tokens (no free-form sentences).
    # Using place5-style detailed prompts with exact regex patterns for maximum recall
    output_columns = {
        "brand": (
            f"Extract the main storage brand from the normalized_name. The minimal set of brands is {brands}. "
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
            "Extract the storage capacity from the normalized_name. CRITICAL FOR RECALL: Capacity is essential for blocking - same brand but different capacity = different product. "
            "Missing capacity causes significant recall loss. "
            "Use regex over the normalized_name to detect patterns like '32 gb', '64gb', '128 go', '1tb', '2 tb', '256gb', '512 gb', '1 tb', '128', '256', '512'. "
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
            "1. First, search for regex r'xqd|ssd|tv|lte' in the normalized_name. "
            "2. If 'lte' is found, return 'phone' (not 'lte'). "
            "3. If no match from step 1, check if 'fdrive' is in the text (case-insensitive) - if yes, return 'fdrive'. "
            "4. If still no match, check if 'memcard' is in the text (case-insensitive) - if yes, return 'memcard'. "
            "5. If a match from step 1 exists and it's not 'lte', return that match (e.g., 'xqd', 'ssd', 'tv'). "
            "6. Otherwise return '0'. "
            "Return single lowercase token only."
        ),
        "type": (
            f"Extract a short product type / line relative to the brand, based on the family keywords mapping {families}, "
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
        "item_code": (
            "Extract explicit numeric item codes, especially those in parentheses, using regex like r'\\((mk)?[0-9]{6,10}\\)'. "
            "Strip parentheses and any 'mk' prefix, leaving only the digits. "
            "Also look for patterns like '(4187407)', '(mk483394661)', '(173473)', etc. "
            "RELAXED: Also extract codes with 4-5 digits if they appear in parentheses. "
            "If multiple codes appear, pick the one most likely to be a manufacturer part number (usually longer, in parentheses). "
            "Return the digits as a string, or '0' if none found."
        ),
        "series": (
            f"Extract the product series / family token. Use the brand-specific families mapping {families} "
            f"and the Intenso-specific list {intenso_type} and Samsung color names {colors} over the normalized_name. "
            "Normalize obvious typos ('cruizer' -> 'cruzer'). "
            "Use re.findall() to get ALL matches, deduplicate using set(), sort, join with single space. "
            "Return lowercase family tokens (e.g. 'glide', 'cruzer', 'ultimate', 'exceria', 'jumpdrive', 'premium', 'basic'), "
            "or '0' if none."
        ),
        "pat_hb": (
            "Extract ALL hyphenated alphanumeric patterns, using regex r'\\w+-\\w+' over the normalized_name. "
            "Use re.findall() to get ALL matches, not just the first. "
            "Typical examples are 'uhs-i', 'class-10', 'high-speed', 'type-c', 'usb-3', 'micro-sd'. "
            "Deduplicate using set(), sort, join with single space. "
            "Return all such patterns in lowercase, or '0' if none."
        ),
        "hybrid": (
            "Extract all 'hybrid' identifiers that mix letters and digits and are at least length 5, similar to the "
            "regex r'(?=[^\\W\\d_]*\\d)(?=\\d*[^\\W\\d_])[^\\W_gGM]{5,}'. "
            "This matches alphanumeric codes like 'dt101g2', 'sda10', 'usm32gqx', 'lsd16gcrbeu1000', etc. "
            "Use regex on the normalized_name, deduplicate matches, sort them lexicographically, and join with a single space. "
            "Return '0' if there are no such identifiers."
        ),
        "long_num": (
            "Extract all sequences of 4 or more digits using regex r'[0-9]{4,}' on the normalized_name. "
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
            "\n\nCRITICAL: Use the normalized_name column that was already extracted in the previous step. "
            "Do NOT re-normalize - use the existing normalized_name directly for all feature extractions. "
            "\n\nEXTRACTION REQUIREMENTS:\n"
            "1. Follow the regex patterns and cleaning logic provided in each output column description EXACTLY. "
            "2. The patterns are from proven blocking code - use them VERBATIM, do NOT modify or simplify them. "
            "3. For each field, follow the EXACT extraction logic described (findall vs search, deduplication method, sorting, joining). "
            "4. Pre-compile ALL regex patterns using re.compile() for performance. "
            "5. Use the specified methods: re.findall() when description says 'find all', re.search() when it says 'first match'. "
            "6. For deduplication: use set() then convert back to sorted list. "
            "7. For joining: use single space ' '.join(). "
            "8. Return '0' (string zero) for missing values, NOT empty string or None. "
            "\n\nCRITICAL FOR RECALL - HIGHEST PRIORITY:\n"
            "Aim for MAXIMUM RECALL - it's MUCH better to extract something (even if slightly imperfect) than to return '0'. "
            "For blocking/entity matching, missing a feature (returning '0') causes missed matches and DRASTICALLY reduces recall. "
            "If a pattern is even REMOTELY close to matching, extract it. Only return '0' when you're ABSOLUTELY CERTAIN there's no match. "
            "Handle multilingual variations EXTREMELY aggressively - extract when in doubt, extract partial matches, extract variations. "
            "PREFER extracting multiple candidates over returning '0'. If unsure between two values, extract both (space-separated). "
            "For blocking, false positives are acceptable but false negatives (missing features) are catastrophic for recall. "
            "\n\nPERFORMANCE: Optimize for throughput - must handle millions of rows within 15 minutes. "
            "Pre-compile regex patterns, avoid repeated compilation, use efficient string operations. "
            "\n\nIn comments, explain your code and why you chose the approach you did."
        ),
        input_columns=["normalized_name"],
        name="extract_x2_features",
        output_columns=output_columns,
        generate_via_code=True,
    )

    extracted_features = data_ref.skb.eval()
    required_cols = [
        "brand",
        "capacity",
        "normalized_name",
        "mem_type",
        "type",
        "model",
        "item_code",
        "series",
        "pat_hb",
        "hybrid",
        "long_num",
    ]
    for col in required_cols:
        if col not in extracted_features.columns:
            extracted_features[col] = "0"

    for col in required_cols:
        extracted_features[col] = extracted_features[col].fillna("0").astype(str).str.lower()

    # Lowercase string-like columns (but keep non-string / structured cols as-is).
    for col in extracted_features.columns:
        if col == "id":
            continue
        series = extracted_features[col]
        extracted_features[col] = series.astype(str).str.lower()

    return extracted_features
    families = {
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
            "jumpdrive",
            "usb",
            "memo",
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
            "exceria",
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
            "micro",
            "line",
            "scheda",
            "usb",
            "sd",
            "premium",
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
        "0": [
            "adapt",
            "alu",
            "attach",
            "blade",
            "canvas",
            "cart",
            "cruzer",
            "cs/ultra",
            "datatravel",
            "evo",
            "exceria",
            "extern",
            "extreme",
            "flair",
            "flash",
            "galaxy",
            "glide",
            "hyperx",
            "jumpdrive",
            "kart",
            "klasse",
            "line",
            "memo",
            "memoria",
            "multi",
            "origin",
            "pendrive",
            "premium",
            "react",
            "scheda",
            "secure",
            "select",
            "serie",
            "speicher",
            "tarjeta",
            "transmemo",
            "transmemory",
            "traveler",
            "ultimate",
            "ultra",
            "usb",
            "usm32gqx",
            "veloc",
            "wex",
            "xqd",
        ],
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

    # Second: extract blocking features using deterministic regex-heavy code.
    # These columns are fed into `EntityBlocking.block_x2` and used in equality
    # comparisons, so keep them short, canonical tokens (no free-form sentences).
    # Using place5-style detailed prompts with exact regex patterns for maximum recall
    output_columns = {
        "brand": (
            f"Extract the main storage brand from the name. The minimal set of brands is {brands}. "
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
            f"Extract a short product type / line relative to the brand, based on the family keywords mapping {families}, "
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
        "item_code": (
            "Extract explicit numeric item codes, especially those in parentheses, using regex like r'\\((mk)?[0-9]{6,10}\\)'. "
            "Strip parentheses and any 'mk' prefix, leaving only the digits. "
            "Also look for patterns like '(4187407)', '(mk483394661)', '(173473)', etc. "
            "RELAXED: Also extract codes with 4-5 digits if they appear in parentheses. "
            "If multiple codes appear, pick the one most likely to be a manufacturer part number (usually longer, in parentheses). "
            "Return the digits as a string, or '0' if none found."
        ),
        "series": (
            f"Extract the product series / family token. Use the brand-specific families mapping {families} "
            f"and the Intenso-specific list {intenso_type} and Samsung color names {colors} over the normalized name. "
            "Normalize obvious typos ('cruizer' -> 'cruzer'). "
            "Use re.findall() to get ALL matches, deduplicate using set(), sort, join with single space. "
            "Return lowercase family tokens (e.g. 'glide', 'cruzer', 'ultimate', 'exceria', 'jumpdrive', 'premium', 'basic'), "
            "or '0' if none."
        ),
        "pat_hb": (
            "Extract ALL hyphenated alphanumeric patterns, using regex r'\\w+-\\w+' over the normalized name. "
            "Use re.findall() to get ALL matches, not just the first. "
            "Typical examples are 'uhs-i', 'class-10', 'high-speed', 'type-c', 'usb-3', 'micro-sd'. "
            "Deduplicate using set(), sort, join with single space. "
            "Return all such patterns in lowercase, or '0' if none."
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

    extracted_features = data_ref.skb.eval()
    required_cols = [
        "brand",
        "capacity",
        "normalized_name",
        "mem_type",
        "type",
        "model",
        "item_code",
        "series",
        "pat_hb",
        "hybrid",
        "long_num",
    ]
    for col in required_cols:
        if col not in extracted_features.columns:
            extracted_features[col] = "0"

    for col in required_cols:
        extracted_features[col] = extracted_features[col].fillna("0").astype(str).str.lower()

    # Lowercase string-like columns (but keep non-string / structured cols as-is).
    for col in extracted_features.columns:
        if col == "id":
            continue
        series = extracted_features[col]
        extracted_features[col] = series.astype(str).str.lower()

    return extracted_features
