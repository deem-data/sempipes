import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from litellm import completion_cost

import sempipes


def load_sounds_dataset(dataset_size: str = "1000"):
    data_path = f"tests/data/sounds-dataset-{dataset_size}/sounds.csv"
    df = pd.read_csv(data_path, engine="python", on_bad_lines="skip", encoding="utf-8")
    return df


def compute_accuracy(original: pd.Series, extracted: pd.Series) -> float:
    aligned = pd.DataFrame({"original": original, "extracted": extracted}).dropna()

    if len(aligned) == 0:
        return 0.0

    original_normalized = aligned["original"].astype(str).str.strip().str.lower()
    extracted_normalized = aligned["extracted"].astype(str).str.strip().str.lower()

    matches = (original_normalized == extracted_normalized).sum()
    return matches / len(aligned)


@contextmanager
def track_llm_extraction_cost():
    """
    Context manager to track costs from litellm batch_completion calls during LLM-based extraction.
    Returns a list that accumulates costs.
    """
    costs = []

    # Intercept at the sempipes.llm.llm module level where batch_completion is actually used
    from litellm import batch_completion as original_batch_completion

    import sempipes.llm.llm as llm_module

    def tracked_batch_completion(*args, **kwargs):
        responses = original_batch_completion(*args, **kwargs)
        try:
            # batch_completion returns a list of responses
            # Calculate cost for each response in the batch
            batch_cost = 0.0
            for response in responses:
                cost = completion_cost(completion_response=response)
                if cost is not None:
                    batch_cost += float(cost)
            costs.append(batch_cost)
        except Exception as e:
            print(f"Warning: Could not calculate cost: {e}")
        return responses

    # Intercept batch_completion in the sempipes.llm.llm module where it's imported
    original_module_batch_completion = llm_module.batch_completion
    llm_module.batch_completion = tracked_batch_completion

    try:
        yield costs
    finally:
        llm_module.batch_completion = original_module_batch_completion


def extract_category_with_sempipes(df: pd.DataFrame, audio_column: str = "filename") -> tuple[pd.DataFrame, float]:
    """
    Extract category in a single sem_extract_features call using sempipes.

    Args:
        df: DataFrame containing the sounds dataset
        audio_column: Name of the column containing audio file paths

    Returns:
        Tuple of (DataFrame with extracted_category column added, total cost in USD for LLM extraction)
    """
    # Filter out NaN/empty values and check if files exist
    valid_mask = df[audio_column].notna() & (df[audio_column].astype(str).str.strip() != "")
    df_filtered = df[valid_mask].copy()

    valid_audio_mask = df_filtered[audio_column].apply(lambda x: Path(str(x)).exists())
    df_minimal = df_filtered[valid_audio_mask].copy()

    if len(df_minimal) == 0:
        result_df = df.copy()
        result_df["extracted_category"] = "Unknown"
        return result_df, 0.0

    df_minimal = df_minimal[[audio_column]].copy().reset_index()
    original_index_name = df.index.name if df.index.name is not None else "index"

    data_op = skrub.var("sounds_data", df_minimal)

    # Get unique category values for the prompt
    unique_values = (
        sorted([str(v) for v in df["category"].dropna().unique() if pd.notna(v)]) if "category" in df.columns else []
    )
    categories_list = ", ".join([f'"{cat}"' for cat in unique_values])

    output_columns = {
        "extracted_category": f"""Extract the category of the environmental sound. The possible categories are: {categories_list}.\n\nReturn only the category name as a single string value."""
    }

    nl_prompt = """You are a helpful audio environmental sound classification expert. Analyze each environmental sound audio file and classify it into the appropriate category.

IMPORTANT: You are receiving audio files as base64-encoded data URLs. You must process and analyze the actual audio content to classify the environmental sound. Listen to the audio and identify the sound category based on what you hear."""

    # WARNING: Most LLM models (including Gemini) do not support audio input in the format used by sempipes.
    # If accuracy is very low (<0.1), the model is likely not processing the audio correctly.
    # Consider using code generation mode (generate_via_code=True) instead, which uses audio processing libraries.
    print("WARNING: LLM-based audio extraction may not work correctly if the model doesn't support audio input.")
    print("If accuracy is very low, consider using sempipes_audio_extraction.py (code generation mode) instead.")

    with track_llm_extraction_cost() as costs:
        result_op = data_op.sem_extract_features(
            nl_prompt=nl_prompt,
            input_columns=[audio_column],
            name="extract_category",
            output_columns=output_columns,
            generate_via_code=False,
        )
        result_df_minimal = result_op.skb.eval()

    total_cost = sum(costs) if costs else 0.0
    result_df = df.copy().reset_index()

    extracted_col_name = "extracted_category"
    if extracted_col_name in result_df_minimal.columns:
        merge_cols = [original_index_name, extracted_col_name]
        if original_index_name in result_df_minimal.columns:
            result_df = result_df.merge(result_df_minimal[merge_cols], on=original_index_name, how="left")
        else:
            extracted_values = result_df_minimal[extracted_col_name].values
            if len(extracted_values) <= len(result_df):
                result_df[extracted_col_name] = None
                result_df.loc[: len(extracted_values) - 1, extracted_col_name] = extracted_values
            else:
                result_df[extracted_col_name] = extracted_values[: len(result_df)]

        result_df[extracted_col_name] = result_df[extracted_col_name].fillna("Unknown")
    else:
        result_df[extracted_col_name] = "Unknown"

    if original_index_name in result_df.columns:
        result_df = result_df.set_index(original_index_name)
        result_df.index.name = df.index.name

    print("\nExtracted category from audio in a single sem_extract_features call")
    print(f"  LLM extraction cost: ${total_cost:.6f}")

    if "category" in df.columns and extracted_col_name in result_df.columns:
        accuracy = compute_accuracy(df["category"], result_df[extracted_col_name])
        print(f"  Accuracy for category: {accuracy:.3f} ({accuracy*100:.2f}%)")

    return result_df, total_cost


def extract_with_repeats_sempipes(
    df: pd.DataFrame, audio_column: str = "filename", num_repeats: int = 3
) -> tuple[list[pd.DataFrame], list[float], float]:
    """
    Extract category multiple times using sempipes and compute accuracy statistics.

    Args:
        df: DataFrame containing the sounds dataset
        audio_column: Name of the column containing audio file paths
        num_repeats: Number of times to repeat the extraction

    Returns:
        Tuple of (list of result DataFrames, list of accuracies, total cost across all repeats)
    """
    accuracies = []
    all_results = []
    total_cost = 0.0

    for repeat_num in range(num_repeats):
        print(f"\n{'='*80}")
        print(f"REPEAT {repeat_num + 1} of {num_repeats}")
        print(f"{'='*80}")

        result_df, repeat_cost = extract_category_with_sempipes(df, audio_column)

        total_cost += repeat_cost

        df = df.sort_values(by="filename").reset_index(drop=True)
        result_df = result_df.sort_values(by="filename").reset_index(drop=True)

        # Debug prints and sum matches calculation
        if "category" in df.columns and "extracted_category" in result_df.columns:
            print(df[["filename", "category"]])
            print(result_df[["filename", "extracted_category"]])

            sum_matches = sum(
                df["category"].astype(str).str.strip().str.lower()
                == result_df["extracted_category"].astype(str).str.strip().str.lower()
            )
            print(f"Sum of matches: {sum_matches}, accuracy {sum_matches / len(df):.3f}")

        accuracy = compute_accuracy(df["category"], result_df["extracted_category"])
        accuracies.append(accuracy)
        print(f"\n  Accuracy (repeat {repeat_num + 1}): {accuracy:.3f} ({accuracy*100:.2f}%)")

        all_results.append(result_df)

    return all_results, accuracies, total_cost


if __name__ == "__main__":
    sempipes.update_config(batch_size_for_batch_processing=30)
    sempipes.update_config(llm_for_batch_processing=sempipes.LLM(name="gemini-2.5-flash"))

    gemini_credentials_path = "/Users/olgaovcharenko/.config/gcloud/application_default_credentials.json"
    os.environ["GEMINI_CREDENTIALS_PATH"] = gemini_credentials_path
    os.environ["VERTEXAI_PROJECT"] = "bq-mm-benchmark"
    os.environ["VERTEXAI_LOCATION"] = "us-central1"

    df = load_sounds_dataset("10000")
    print(f"\nLoaded dataset with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    num_repeats = 1
    try:
        all_results, accuracies, total_cost = extract_with_repeats_sempipes(
            df, audio_column="filename", num_repeats=num_repeats
        )

        result_df = all_results[-1]

        print(f"\nOriginal columns: {list(df.columns)}")
        print(f"New extracted columns: {[col for col in result_df.columns if col not in df.columns]}")

        average_accuracy = np.round(np.mean(accuracies), 3) if accuracies else 0.0
        average_accuracy_std = np.round(np.std(accuracies), 3) if accuracies else 0.0
        print(f"\nAverage accuracy across all repeats: {average_accuracy} +- {average_accuracy_std}")

        print("\nCOST SUMMARY (LLM Extraction)")
        print(f"{'='*80}")
        print(f"  Total cost across all repeats: ${total_cost:.6f}")
        print(f"  Average cost per repeat: ${total_cost / num_repeats:.6f}")

        output_path = "experiments/feature_extraction/extracted_features_sempipes_sounds_llm.csv"
        result_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Feature extraction complete!")
