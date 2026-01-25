import os

import numpy as np
import palimpzest as pz
import pandas as pd
from dotenv import load_dotenv
from palimpzest.constants import Model

from experiments.feature_extraction.text_legal.sempipes_text_extraction import compute_accuracy

# Define the schema for the contracts dataset
data_cols = [
    {"name": "clause_text", "type": str, "desc": "The legal financial clause text."},
    {"name": "id", "type": str, "desc": "The id of the financial clause."},
]


class ContractsDataset(pz.IterDataset):
    def __init__(self, id: str, contracts_df: pd.DataFrame):
        super().__init__(id=id, schema=data_cols)
        self.contracts_df = contracts_df

    def __len__(self):
        return len(self.contracts_df)

    def __getitem__(self, idx: int):
        # get row from dataframe
        return self.contracts_df.iloc[idx].to_dict()


def run(pz_config: pz.QueryProcessorConfig, contracts_df: pd.DataFrame):
    # Find the text column (could be "text", "clause", "contract_text", etc.)
    text_column = "clause_text"

    # Create dataset
    contracts_data = ContractsDataset(id="contracts-data", contracts_df=contracts_df[[text_column, "id"]])

    # Get unique clause_type values from the dataset to help the LLM
    unique_clause_types = sorted(contracts_df["clause_type"].dropna().unique().tolist())
    clause_types_list = ", ".join([f'"{ct}"' for ct in unique_clause_types])

    clause_type_desc = f"""Classify the financial clause into its specific type. Analyze the legal text carefully and identify the clause type. The clause type should be one of the following: {clause_types_list}."""
    clause_type_desc += "\n\nReturn only the clause type name as a single string value."

    contracts_data = contracts_data.sem_map(
        [
            {"name": "extracted_clause_type", "type": str, "desc": clause_type_desc, "depends_on": [text_column]},
        ]
    )

    output = contracts_data.run(pz_config)
    cost = output.execution_stats.total_execution_cost
    output_df = output.to_df()[["extracted_clause_type", "id"]]
    return output_df, cost


# Load data
# load_dotenv()
dataset_size = "10000"
csv_path = f"tests/data/contracts-dataset-{dataset_size}/contracts.csv"
contracts_df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8")

print(f"\nLoaded dataset with {len(contracts_df)} rows")
print(f"Columns: {list(contracts_df.columns)}")

column_name = "clause_type"
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
        gemini_credentials_path="/Users/olgaovcharenko/.config/gcloud/application_default_credentials.json",
    )

    result, cost = run(pz_config, contracts_df)
    result.to_csv(f"pz_text_extraction_result_{repeat_num + 1}.csv")

    total_costs.append(cost)
    all_results.append(result)

    contracts_df = contracts_df.sort_values(by="id").reset_index(drop=True)
    result = result.sort_values(by="id").reset_index(drop=True)

    print(contracts_df[["id", "clause_type"]])
    print(result[["id", "extracted_clause_type"]])

    sum_matches = sum(
        contracts_df[column_name].astype(str).str.strip().str.lower()
        == result["extracted_clause_type"].astype(str).str.strip().str.lower()
    )
    print(f"Sum of matches: {sum_matches}, accuracy {sum_matches / len(contracts_df):.4f}")

    accuracy = compute_accuracy(contracts_df[column_name], result["extracted_clause_type"])
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
