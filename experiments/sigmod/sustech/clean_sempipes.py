import re

import pandas as pd
import skrub

import sempipes

brands = [
    "compaq",
    "toshiba",
    "sony",
    "ibm",
    "epson",
    "xmg",
    "vaio",
    "samsung",
    "panasonic ",
    "nec ",
    "gateway",
    "google",
    "fujitsu",
    "eurocom",
    "asus",
    "alienware",
    "dell",
    "aoson",
    "gemei",
    "msi",
    "lenovo",
    "acer",
    "asus",
    "hp",
    "lg ",
    "microsoft",
    "apple",
]

cpu_brands = ["intel", "amd"]

intel_cores = [" i3", " i5", " i7", "2 duo", "celeron", "pentium", "centrino"]
amd_cores = ["e-series", "a8", "radeon", "athlon", "turion", "phenom"]

families = {
    "hp": [r"elitebook", r"compaq", r"folio", r"pavilion"],
    "lenovo": [r" x[0-9]{3}[t]?", r"x1 carbon"],
    "dell": [r"inspiron"],
    "asus": [
        r"zenbook",
    ],
    "acer": [
        r"aspire",
        r"extensa",
    ],
    "0": [],
}

families_brand = {
    "elitebook": {"hp", "panasonic "},  # panasonic
    "compaq": {"hp"},
    "folio": {"hp"},
    "pavilion": {"hp", "panasonic "},  # panasonic
    "inspiron": {"dell", "lenovo"},  # lenovo
    "zenbook": {"asus"},
    "aspire": {"acer"},
    "extensa": {"acer"},
    "thinkpad": {"lenovo"},  # panasonic
    "thinkcentre": {"lenovo"},
    "thinkserver": {"lenovo"},
    "toughbook": {"panasonic "},
    "envy": {"hp"},
    "macbook": {"apple"},
    "probook": {"hp"},
    "latitude": {"dell"},
    # 'chromebook': {'0'},
    "tecra": {"toshiba"},
    "touchsmart": {"hp"},
    # 'dominator':'msi',
    "satellite": {"toshiba"},
}


def to_brand_dict(x):
    if isinstance(x, dict):
        return x  # already in the right form
    if pd.isna(x):
        return {}  # no brand detected
    # if x is like "dell,hp", split; otherwise single brand
    parts = [p.strip() for p in str(x).split(",") if p.strip()]
    return {p: None for p in parts}


def clean_sempipes_naive(data) -> pd.DataFrame:
    """
    Clean X1.csv data to a readable format.

    :param data: X1.csv
    :return:
        A DataFrame which contains following columns:
        {instance_id: instance_id of items;
         brand: computer's brand, range in: {'dell', 'lenovo', 'acer', 'asus', 'hp'};
         cpu_brand: cpu's brand, range in: {'intel', 'amd'};
         cpu_core: cpu extra information, relative to cpu_brand;
         cpu_model: cpu model, relative to cpu_brand;
         cpu_frequency: cpu's frequency, unit in Hz;
         ram_capacity: capacity of RAM, unit in GB;
         display_size: size of computer;
         pc_name: name information extract from title;
         name_family: family name of computer;
         title: title information of instance}
         If the value can't extract from the information given, '0' will be filled.
    """
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="openai/gpt-5.2",
            # parameters={"temperature": 0.8},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="openai/gpt-4.1",
            parameters={"temperature": 0.0},
        ),
    )

    data_ref = skrub.var("data_original_x1", data)

    #  First pass to get generic columns
    output_columns = {
        "brand": f"Extract the brand(s) of the PC from the title. The minimal set of brands is {brands}. You can extend the brands list. Also, utilize this mapping of families to brands: {families_brand}. Please use mutiple combined fast approaches. PC can have multiple brands, return them as a string separated by a comma, e.g., 'toshiba, panasonic'.",
        "cpu_brand": f"Extract the CPU brand. The minimal set of brands is {cpu_brands}. Please extract the CPU brand from the title. Extract only brands from the given list.",
        "cpu_model": "Extract the model(s) of the CPU. Use multiple approaches like regex with units, keywords, and some transformers models. Return them as a string separated by a comma, e.g., 'i5-12500H, i7-12700H'.",
        "cpu_frequency": "Extract the frequency of the CPU. Use multiple approaches like regex with units like [Gg][Hh][Zz], keywords, and some transformers models.",
        "cpu_core": f"Extract the core of the CPU based on the CPU brand and title. Use multiple approaches like regex with units, keywords, and some transformers models. The minimal set of cores is {intel_cores} for Intel and {amd_cores} for AMD.",
        "ram_capacity": "Extract the capacity of the RAM. Use multiple approaches like regex with units, keywords, and some transformers models.",
        "display_size": "Extract the size of the display. Use multiple approaches like regex with units like inch, keywords, and some transformers models.",
        "pc_name": "Extract the PC name from the title. Use multiple approaches like regex, keywords, and some transformers models.",
        "family": f"Extract the brand family, check the extarcted brand and pc name. Use multiple approaches like regex with units, keywords, and some transformers models. The minimal mapping of brands to families is {families}. You can also use the mapping of families to brands: {families_brand} and brand column.",
    }

    data_ref = data_ref.sem_extract_features(
        nl_prompt="YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. Use ONLY regex and rule-based approaches, DO NOT USE ANY Transformer, NER, LLM fallbacks, or LM models. Given dirty titles of PCs from a retailer, extract features useful for deduplication. Analyze data to understand the patterns and use them to extract features. In comments, explain your code and why you chose the approach you did.",
        input_columns=["title"],
        name="extract_features_x1_1",
        output_columns=output_columns,
        generate_via_code=True,
    )

    extracted_features = data_ref.skb.eval()
    print(extracted_features)
    extracted_features.to_csv("extracted_features_x1_1.csv", index=False)

    data_ref = data_ref.sem_extract_features(
        nl_prompt="YOU ARE PROHIBITED TO USE TRANSFORMERS LIBRARY. You are given very dirty titles of PCs from a retailer. Normalize the title by lowercasing, removing special characters, extra spaces, and standardizing common abbreviations and units. Extract a sorted set of all unique alphanumeric tokens (words and numbers) from the title after removing special characters and stopwords. Make title easier for regex matching and deduplication. DO NOT use any LM models.",
        input_columns=["title"],
        name="extract_features_x1_2",
        output_columns={
            "normalized_title": "Clean and normalize the title by lowercasing, removing special characters, extra spaces, and standardizing common abbreviations and units. Extract a sorted set of all unique alphanumeric tokens (words and numbers) from the title after removing special characters and stopwords. Make title easier for regex matching and deduplication."
        },
        generate_via_code=True,
    )

    extracted_features = data_ref.skb.eval()
    print(extracted_features)
    extracted_features.to_csv("extracted_features_x1_2.csv", index=False)

    result = pd.DataFrame(extracted_features)
    # result = pd.read_csv("extracted_features_x1_2.csv")
    name = [
        "instance_id",
        "brand",
        "cpu_brand",
        "cpu_core",
        "cpu_model",
        "cpu_frequency",
        "ram_capacity",
        "display_size",
        "pc_name",
        "family",
        "title",
    ]
    for i in range(len(name)):
        result.rename({i: name[i]}, inplace=True, axis=1)

    # Lowercase string-like columns (but keep non-string / structured cols as-is).
    for col in result.columns:
        if col == "instance_id":
            continue
        series = result[col]
        result[col] = series.astype(str).str.lower()

    result["brand"] = result["brand"].apply(to_brand_dict)
    result["cpu_brand"] = result["cpu_brand"].fillna("0")
    result["cpu_model"] = result["cpu_model"].apply(lambda x: set(x.split(", ")) if isinstance(x, str) else set())

    return result
