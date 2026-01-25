import os

# Set SCIPY_ARRAY_API=1 to enable scipy array API compatibility. Must be set before any scipy/sklearn imports.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import dotenv

dotenv.load_dotenv()

from contextlib import contextmanager  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import skrub  # noqa: E402
from litellm import completion, completion_cost  # noqa: E402

import sempipes  # noqa: E402


def load_contracts_dataset(dataset_size: str = "1000"):
    """Load the contracts dataset."""
    data_path = f"tests/data/contracts-dataset-{dataset_size}/contracts.csv"
    df = pd.read_csv(data_path, engine="python", on_bad_lines="skip", encoding="utf-8")
    return df


def compute_accuracy(original: pd.Series, extracted: pd.Series) -> float:
    """Compute accuracy between original and extracted values."""
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


def extract_clause_type_with_sempipes(df: pd.DataFrame, text_column: str = "clause_text") -> tuple[pd.DataFrame, float]:
    """
    Extract clause_type in a single sem_extract_features call using sempipes.

    Args:
        df: DataFrame containing the contracts dataset
        text_column: Name of the column containing contract/clause text

    Returns:
        Tuple of (DataFrame with extracted_clause_type column added, total cost in USD for LLM extraction)
    """
    # Work on a copy and add a stable row id for alignment in downstream joins
    df = df.copy()
    df["_row_id"] = np.arange(len(df))

    # Filter out NaN/empty values
    valid_mask = df[text_column].notna() & (df[text_column].astype(str).str.strip() != "")
    df_minimal = df[[text_column]].copy()
    df_minimal = df_minimal[valid_mask].copy()

    if len(df_minimal) == 0:
        result_df = df.copy()
        result_df["extracted_clause_type"] = "Unknown"
        return result_df, 0.0

    df_minimal = df_minimal.reset_index()
    original_index_name = df.index.name if df.index.name is not None else "index"

    data_op = skrub.var("contracts_data", df_minimal)

    # Get unique clause types for the prompt
    unique_values = (
        sorted([str(v) for v in df["clause_type"].dropna().unique() if pd.notna(v)])
        if "clause_type" in df.columns
        else []
    )
    clause_types_list = ", ".join([f'"{ct}"' for ct in unique_values])

    output_columns = {
        "extracted_clause_type": f"""Classify the financial clause into its specific type. Analyze the legal text carefully and identify the clause type. The clause type should be one of the following: {clause_types_list}.\n\nReturn only the clause type name as a single string value."""
    }

    nl_prompt = """Classify financial contract clauses into their clause types. Analyze each contract clause text carefully and identify the specific clause type."""

    with track_llm_extraction_cost() as costs:
        result_op = data_op.sem_extract_features(
            nl_prompt=nl_prompt,
            input_columns=[text_column],
            name="extract_clause_type",
            output_columns=output_columns,
            generate_via_code=False,
        )
        result_df_minimal = result_op.skb.eval()

    total_cost = sum(costs) if costs else 0.0
    result_df = df.copy().reset_index()

    extracted_col_name = "extracted_clause_type"
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

    print("\nExtracted clause_type from text in a single sem_extract_features call")
    print(f"  LLM extraction cost: ${total_cost:.6f}")

    if "clause_type" in df.columns and extracted_col_name in result_df.columns:
        accuracy = compute_accuracy(df["clause_type"], result_df[extracted_col_name])
        print(f"  Accuracy for clause_type: {accuracy:.3f} ({accuracy*100:.2f}%)")

    return result_df, total_cost


def extract_with_repeats_sempipes(
    df: pd.DataFrame, text_column: str = "clause_text", num_repeats: int = 3
) -> tuple[list[pd.DataFrame], list[float], float]:
    """
    Extract clause_type multiple times using sempipes and compute accuracy statistics.

    Args:
        df: DataFrame containing the contracts dataset
        text_column: Name of the column containing contract/clause text
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

        result_df, repeat_cost = extract_clause_type_with_sempipes(df, text_column)

        total_cost += repeat_cost

        # Do NOT reorder or sort. Keep the original row order for alignment.
        df = df.reset_index(drop=True)
        result_df = result_df.reset_index(drop=True)

        # Debug prints and sum matches calculation
        if "clause_type" in df.columns and "extracted_clause_type" in result_df.columns:
            preview_cols = ["id", "clause_type"] if "id" in df.columns else ["clause_type"]
            preview_cols_result = (
                ["id", "extracted_clause_type"] if "id" in result_df.columns else ["extracted_clause_type"]
            )
            print(df[preview_cols].head(10))
            print(result_df[preview_cols_result].head(10))

            sum_matches = sum(
                df["clause_type"].astype(str).str.strip().str.lower()
                == result_df["extracted_clause_type"].astype(str).str.strip().str.lower()
            )
            print(f"Sum of matches: {sum_matches}, accuracy {sum_matches / len(df):.3f}")

        accuracy = compute_accuracy(df["clause_type"], result_df["extracted_clause_type"])
        accuracies.append(accuracy)
        print(f"\n  Accuracy (repeat {repeat_num + 1}): {accuracy:.3f} ({accuracy*100:.2f}%)")

        all_results.append(result_df)

    return all_results, accuracies, total_cost


if __name__ == "__main__":
    sempipes.update_config(batch_size_for_batch_processing=10)
    sempipes.update_config(llm_for_batch_processing=sempipes.LLM(name="gemini-2.5-flash"))

    df = load_contracts_dataset("10000")
    print(f"\nLoaded dataset with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    text_column = "clause_text"
    num_repeats = 1
    try:
        all_results, accuracies, total_cost = extract_with_repeats_sempipes(
            df, text_column=text_column, num_repeats=num_repeats
        )
        print("Results=" * 80)
        print(all_results)
        print("DF=" * 80)
        print(df)

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

        output_path = "experiments/feature_extraction/extracted_features_sempipes_contracts.csv"
        result_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Feature extraction complete!")
