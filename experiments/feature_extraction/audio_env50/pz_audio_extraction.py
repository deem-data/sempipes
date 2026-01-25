import os

import numpy as np
import palimpzest as pz
import pandas as pd
from dotenv import load_dotenv
from palimpzest.constants import Model
from palimpzest.core.lib.schemas import AudioFilepath


def compute_accuracy(original: pd.Series, extracted: pd.Series) -> float:
    """
    Compute accuracy between original and extracted values.

    Args:
        original: Series with original/ground truth values
        extracted: Series with extracted values

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    # Align the series and handle missing values
    aligned = pd.DataFrame({"original": original, "extracted": extracted})

    if len(aligned) == 0:
        return 0.0

    # Convert to string and normalize (case-insensitive, strip whitespace)
    original_normalized = aligned["original"].astype(str).str.strip().str.lower()
    print(original_normalized)
    extracted_normalized = aligned["extracted"].astype(str).str.strip().str.lower()

    print("original_normalized")
    print(original_normalized)
    print("extracted_normalized")
    print(extracted_normalized)

    # Compute accuracy
    matches = (original_normalized == extracted_normalized).sum()
    return matches / len(aligned)


# Define the schema for the sounds dataset
data_cols = [
    {"name": "filename", "type": AudioFilepath, "desc": "The filepath containing audio files."},
    {"name": "src_file", "type": str, "desc": "The source file identifier."},
]


class SoundsDataset(pz.IterDataset):
    def __init__(self, id: str, sounds_df: pd.DataFrame):
        super().__init__(id=id, schema=data_cols)
        self.sounds_df = sounds_df

    def __len__(self):
        return len(self.sounds_df)

    def __getitem__(self, idx: int):
        # get row from dataframe
        return self.sounds_df.iloc[idx].to_dict()


def run(pz_config: pz.QueryProcessorConfig, sounds_df: pd.DataFrame):
    # Create dataset
    sounds_data = SoundsDataset(id="sounds-data", sounds_df=sounds_df[["filename", "src_file"]])

    # Get unique category values from the dataset to help the LLM
    unique_categories = (
        sorted(sounds_df["category"].dropna().unique().tolist()) if "category" in sounds_df.columns else []
    )
    categories_list = ", ".join([f'"{cat}"' for cat in unique_categories])

    category_desc = (
        f"""Extract the category of the environmental sound. The possible categories are: {categories_list}."""
    )
    category_desc += "\n\nReturn only the category name as a single string value."

    sounds_data = sounds_data.sem_map(
        [
            {"name": "extracted_category", "type": str, "desc": category_desc, "depends_on": ["filename"]},
        ]
    )

    output = sounds_data.run(pz_config)
    cost = output.execution_stats.total_execution_cost
    output_df = output.to_df()[["extracted_category", "filename"]]
    return output_df, cost


# Load data
load_dotenv()
dataset_size = "1000"
csv_path = f"tests/data/sounds-dataset-{dataset_size}/sounds.csv"
sounds_df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8")

print(f"\nLoaded dataset with {len(sounds_df)} rows")
print(f"Columns: {list(sounds_df.columns)}")

column_name = "category"
num_repeats = 5

# Track results across repeats
all_results = []
overall_accuracies = []
costs_per_repeat = []
total_costs = []

for repeat_num in range(num_repeats):
    print(f"\n{'='*80}")
    print(f"REPEAT {repeat_num + 1} of {num_repeats}")
    print(f"{'='*80}")

    # Create fresh config for each repeat to avoid mutation issues
    pz_config = pz.QueryProcessorConfig(
        verbose=False,
        progress=True,
        available_models=[Model.GEMINI_2_5_FLASH],
    )

    result, cost = run(pz_config, sounds_df)

    sounds_df = sounds_df.sort_values(by="filename").reset_index(drop=True)
    result = result.sort_values(by="filename").reset_index(drop=True)

    print(sounds_df[["filename", "category"]])
    print(result[["filename", "extracted_category"]])

    sum_matches = sum(
        sounds_df[column_name].astype(str).str.strip().str.lower()
        == result["extracted_category"].astype(str).str.strip().str.lower()
    )
    print(f"Sum of matches: {sum_matches}, accuracy {sum_matches / len(sounds_df):.4f}")

    costs_per_repeat.append(cost)
    total_costs.append(cost)
    all_results.append(result)

    accuracy = compute_accuracy(sounds_df[column_name], result["extracted_category"])
    overall_accuracies.append(accuracy)
    print(f"\n Overall accuracy (repeat {repeat_num + 1}): {accuracy:.4f}")

# Print final summary
print(f"\n{'='*80}")
print("FINAL SUMMARY (All Repeats)")
print(f"{'='*80}")
print(
    f"Average accuracy: {np.mean(overall_accuracies):.4f} +- {np.std(overall_accuracies):.4f} ({np.mean(overall_accuracies)*100:.2f}%)"
)
print(f"Total cost (all repeats): ${sum(total_costs):.6f}")
print(f"Average cost per repeat: ${np.mean(total_costs):.6f}")
print(f"{'='*80}\n")
