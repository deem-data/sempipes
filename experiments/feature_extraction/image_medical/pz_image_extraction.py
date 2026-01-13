import os

import numpy as np
import palimpzest as pz
import pandas as pd
from dotenv import load_dotenv
from palimpzest.constants import Model
from palimpzest.core.lib.schemas import ImageFilepath

from experiments.feature_extraction.image_medical.sempipes_image_extraction import compute_accuracy

# define the schema for the chest X-ray medical dataset
chestxray_image_data_cols = [
    {"name": "path", "type": ImageFilepath, "desc": "The filepath containing the chest X-ray image"},
    {"name": "filename", "type": str, "desc": "The filename of the chest X-ray image"},
]


class ChestXRayDataset(pz.IterDataset):
    def __init__(self, id: str, chestxray_df: pd.DataFrame):
        super().__init__(id=id, schema=chestxray_image_data_cols)
        self.chestxray_df = chestxray_df

    def __len__(self):
        return len(self.chestxray_df)

    def __getitem__(self, idx: int):
        # get row from dataframe
        return self.chestxray_df.iloc[idx].to_dict()


def run(pz_config: pz.QueryProcessorConfig, chestxray_df: pd.DataFrame):
    # Create dataset
    dataset = ChestXRayDataset(id="chestxray-data", chestxray_df=chestxray_df[["path", "filename"]])

    # Get unique class values from the dataset to help the LLM
    unique_classes = sorted(chestxray_df["class"].dropna().unique().tolist())
    classes_list = ", ".join([f'"{cls}"' for cls in unique_classes])

    class_desc = f"""You are a medical imaging expert specializing in chest X-ray analysis. Analyze this chest X-ray image and classify it. The possible classes are: {classes_list}"""
    class_desc += "\n\nReturn only the class name as a single string value."

    dataset = dataset.sem_map(
        [
            {"name": "predicted_class", "type": str, "desc": class_desc, "depends_on": ["path"]},
        ]
    )

    output = dataset.run(pz_config)
    cost = output.execution_stats.total_execution_cost
    output_df = output.to_df()[["predicted_class", "filename"]]
    return output_df, cost


# Load data
load_dotenv()
dataset_size = "1000"
csv_path = f"tests/data/chestxray-dataset-{dataset_size}/chestxray.csv"
chestxray_df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8")

print(f"\nLoaded dataset with {len(chestxray_df)} rows")
print(f"Columns: {list(chestxray_df.columns)}")

column_name = "class"
num_repeats = 5

# Track results across repeats
all_results = []
overall_accuracies = []
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

    result, cost = run(pz_config, chestxray_df)

    chestxray_df = chestxray_df.sort_values(by="filename").reset_index(drop=True)
    result = result.sort_values(by="filename").reset_index(drop=True)

    print(chestxray_df[["filename", "class"]])
    print(result[["filename", "predicted_class"]])

    sum_matches = sum(
        chestxray_df[column_name].astype(str).str.strip().str.lower()
        == result["predicted_class"].astype(str).str.strip().str.lower()
    )
    print(f"Sum of matches: {sum_matches}, accuracy {sum_matches / len(chestxray_df):.4f}")

    total_costs.append(cost)
    all_results.append(result)

    accuracy = compute_accuracy(chestxray_df[column_name], result["predicted_class"])
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
