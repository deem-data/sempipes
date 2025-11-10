import pandas as pd
import skrub

import sempipes  # pylint: disable=unused-import
from sempipes.config import ensure_default_config


def test_sem_clean_code_generation():
    ensure_default_config()
    df = pd.DataFrame(
        {"name": [" Alice ", "bob", None, "Charlie  ", "Martina"], "age": ["25", "30 ", "NA", "40", "150"]}
    )

    dop = skrub.var("df", df)

    # Use code generation mode (will be skipped quickly if LLM not configured)
    cleaned = dop.sem_clean(
        nl_prompt="Clean the data. For name strip whitespace from name, capitalize. For age, coerce age to integers when possible, impute invalid ages, remove outliers. If there are any NaN, NA, None values in both columns, mark them as 'NaN', do not convert valid values.",
        columns=["name", "age"],
    ).skb.eval()

    print(cleaned)

    assert cleaned["name"].tolist() == ["Alice", "Bob", "NaN", "Charlie", "Martina"]
    assert cleaned["age"].le(50).all()
