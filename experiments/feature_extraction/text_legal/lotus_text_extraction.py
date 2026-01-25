#!/usr/bin/env python3
import os

import lotus
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lotus.models import LM

from experiments.feature_extraction.text_legal.sempipes_text_extraction import compute_accuracy

PRICING = {
    "gpt-4o": {"text": 2.5, "audio": 2.5, "output": 10.0},
    "gpt-4o-mini": {"text": 0.15, "audio": 0.15, "output": 0.6},
    "gpt-4o-audio-preview": {
        "text": 2.5,
        "audio": 2.5,
        "output": 10.0,
    },
    "gpt-5": {"text": 1.25, "audio": 1.25, "output": 10.0},
    "gpt-5-mini": {"text": 0.25, "audio": 0.25, "output": 2.0},
    "gemini-2.0-flash": {
        "text": 0.15,
        "audio": 1.0,
        "output": 0.6,
    },
    "gemini-2.5-flash": {
        "text": 0.3,
        "audio": 1.0,
        "output": 2.5,
    },
    "gemini-2.5-flash-lite": {
        "text": 0.1,
        "audio": 0.3,
        "output": 0.4,
    },
    "gemini-2.5-pro": {
        "text": 1.25,
        "audio": 1.25,
        "output": 10.0,
    },
}


def _calculate_cost(prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
    """Calculate cost based on token usage and model pricing."""
    model_name_lower = model_name.lower()

    # Find matching pricing configuration
    pricing_config = None
    for model_key in PRICING:
        if model_key.lower() in model_name_lower:
            pricing_config = PRICING[model_key]
            break

    if pricing_config is None:
        print(f"Warning: No pricing found for model '{model_name}', cost calculation skipped")
        return 0.0

    # Calculate cost: prices are per 1M tokens
    input_cost = (prompt_tokens / 1_000_000) * pricing_config["text"]
    output_cost = (completion_tokens / 1_000_000) * pricing_config["output"]
    total_cost = input_cost + output_cost

    return total_cost


def _update_token_usage():
    try:
        # Get usage stats from LOTUS using the correct API
        prompt_tokens = lotus.settings.lm.stats.physical_usage.prompt_tokens
        completion_tokens = lotus.settings.lm.stats.physical_usage.completion_tokens
        total_tokens = lotus.settings.lm.stats.physical_usage.total_tokens
        total_cost = lotus.settings.lm.stats.physical_usage.total_cost

        # Get model name for cost calculation
        model_name = lotus.settings.lm.model if hasattr(lotus.settings.lm, "model") else "gemini-2.5-flash"

        # Calculate cost using our token consumption
        calculated_cost = _calculate_cost(prompt_tokens, completion_tokens, model_name)

        # Print both tokens for comparison
        print(f"prompt token: {prompt_tokens}, completion token: {completion_tokens}")
        print(f"total tokens: {total_tokens}")

        # Print both costs for comparison
        print(f"  LOTUS cost: ${total_cost:.6f}")
        print(f"  Calculated cost based on token consumption: ${calculated_cost:.6f}")

        return total_tokens, calculated_cost

    except Exception as e:
        print(f"Warning: Could not get token usage: {e}")
        return 0.0, 0.0


def run(contracts_df: pd.DataFrame):
    text_column = "clause_text"

    # Get unique clause_type values from the dataset to help the LLM
    unique_clause_types = (
        sorted(contracts_df["clause_type"].dropna().unique().tolist()) if "clause_type" in contracts_df.columns else []
    )
    clause_types_list = ", ".join([f'"{ct}"' for ct in unique_clause_types])

    clause_type_desc = f"""Classify the financial clause into its specific type. Analyze the legal text carefully and identify the clause type. The clause type should be one of the following: {clause_types_list}."""
    clause_type_desc += "\n\nReturn only the clause type name as a single string value."

    contracts_texts = contracts_df[[text_column, "id"]].copy()
    print(contracts_texts)
    clause_types = contracts_texts.sem_extract(
        input_cols=["clause_text"],
        output_cols={"extracted_clause_type": clause_type_desc},
    )

    total_tokens, calculated_cost = _update_token_usage()
    return clause_types[["extracted_clause_type", "id"]], calculated_cost


# Load data
load_dotenv()
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

    lm = LM(
        model="gemini-2.5-flash",
        rate_limit=1000,  # Reduced rate limit to avoid overload
        rate_limit_period=60,
        reasoning_effort="disable",
        max_tokens=512,
        num_retries=2,  # Number of retry attempts for transient errors (503, rate limits, etc.)
        timeout=30,  # Timeout in seconds for each request,
        max_batch_size=10,  # Maximum number of requests to send at once
    )
    lotus.settings.configure(lm=lm)

    result, cost = run(contracts_df)

    contracts_df = contracts_df.sort_values(by="id").reset_index(drop=True)
    result = result.sort_values(by="id").reset_index(drop=True)

    total_costs.append(cost)
    all_results.append(result)

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
