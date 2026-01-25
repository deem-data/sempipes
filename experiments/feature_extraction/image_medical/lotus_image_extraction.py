#!/usr/bin/env python3
import os

import lotus
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lotus.dtype_extensions import ImageArray
from lotus.models import LM

from experiments.feature_extraction.image_medical.sempipes_image_extraction import compute_accuracy

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
    """Get token usage stats from LOTUS and calculate cost."""
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
        return 0, 0.0


def run(chestxray_df: pd.DataFrame):
    """Run image classification on chest X-ray images."""
    # Get unique class values from the dataset
    unique_classes = sorted(chestxray_df["class"].dropna().unique().tolist()) if "class" in chestxray_df.columns else []
    classes_list = ", ".join([f'"{cls}"' for cls in unique_classes])

    class_desc = f"""You are a medical imaging expert specializing in chest X-ray analysis. Analyze this chest X-ray image and classify it. The possible classes are: {classes_list}"""
    class_desc += "\n\nReturn only the class name as a single string value."

    # Create ImageArray from file paths
    image_df = chestxray_df[["path", "filename"]].copy()
    image_df["images"] = ImageArray(
        image_df["path"].apply(lambda p: p if os.path.isabs(p) else os.path.join(os.getcwd(), p))
    )

    # Extract class predictions using sem_extract
    result = image_df.sem_extract(
        input_cols=["images"],
        output_cols={"extracted_class": class_desc},
    )

    total_tokens, calculated_cost = _update_token_usage()
    return result[["extracted_class", "filename"]], calculated_cost


# Load data
load_dotenv()
dataset_size = "10000"
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

    lm = LM(
        model="gemini-2.5-flash",
        rate_limit=1000,  # Reduced rate limit to avoid overload
        rate_limit_period=60,
        reasoning_effort="disable",
        max_tokens=512,
        num_retries=2,  # Number of retry attempts for transient errors (503, rate limits, etc.)
        timeout=30,  # Timeout in seconds for each request
        max_batch_size=10,  # Maximum number of requests to send at once
    )
    lotus.settings.configure(lm=lm)

    result, cost = run(chestxray_df)

    chestxray_df = chestxray_df.sort_values(by="filename").reset_index(drop=True)
    result = result.sort_values(by="filename").reset_index(drop=True)

    total_costs.append(cost)
    all_results.append(result)

    accuracy = compute_accuracy(chestxray_df[column_name], result["extracted_class"])
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
