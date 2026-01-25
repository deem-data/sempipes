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
def track_code_generation_cost():
    """
    Context manager to track costs from litellm completion calls during code generation.
    Returns a list that accumulates costs.
    """
    costs = []

    from litellm import completion as original_completion

    import sempipes.llm.llm as llm_module

    def tracked_completion(*args, **kwargs):
        response = original_completion(*args, **kwargs)
        try:
            cost = completion_cost(completion_response=response)
            costs.append(cost)
        except Exception as e:
            print(f"Warning: Could not calculate cost: {e}")
        return response

    original_module_completion = llm_module.completion
    llm_module.completion = tracked_completion

    try:
        yield costs
    finally:
        llm_module.completion = original_module_completion


def extract_clause_type_with_sempipes(df: pd.DataFrame, text_column: str = "clause_text") -> tuple[pd.DataFrame, float]:
    """
    Extract clause_type in a single sem_extract_features call using sempipes.

    Args:
        df: DataFrame containing the contracts dataset
        text_column: Name of the column containing contract/clause text

    Returns:
        Tuple of (DataFrame with extracted_clause_type column added, total cost in USD for code generation)
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
        "extracted_clause_type": f"""Classify contract clauses from `{text_column}` column into their clause types using zero-shot text classification.

DATASET CONTEXT: This is the contracts-clauses-datasets (mohammedalrashidan/contracts-clauses-datasets) containing legal/financial contract clauses. Dataset: https://www.kaggle.com/datasets/mohammedalrashidan/contracts-clauses-datasets. The task is to classify each clause into its specific type from the provided list.

REQUIRED STEPS:
1. IMPORT: from transformers import pipeline
   NOTE: If you need to use AutoModel classes, the correct name is AutoModelForSeq2SeqLM (NOT AutoModelForSequence2SeqLM).
2. EXTRACT CANDIDATE LABELS: The clause types are: {clause_types_list}. Create candidate_labels = [{', '.join([f'"{ct}"' for ct in unique_values])}]
   Labels are already human-readable. For best results, consider enhancing labels with mini-definitions and synonyms (e.g., "Confidentiality: non-disclosure obligations, permitted disclosures, duration"; "Force majeure: events beyond control, impossibility") as NLI-based zero-shot models score entailment much better with descriptive labels.
3. INITIALIZE CLASSIFIER: Use zero-shot-classification pipeline. Look for high-quality zero-shot classification models on Hugging Face. Prioritize models that:
   - Are based on DeBERTa-v3 architecture (especially large variants) and specifically designed for zero-shot classification tasks
   - Are trained on NLI (Natural Language Inference) datasets like MNLI, SNLI, or XNLI
   - Have "zeroshot" or "zero-shot" in their model identifier
   - Are from reputable organizations or researchers known for NLI work
   - If multilingual text is detected, prioritize models trained on cross-lingual NLI datasets
   
   Use: classifier = pipeline('zero-shot-classification', model=MODEL_NAME, hypothesis_template="This clause is about {{}}.")
   Test multiple hypothesis templates: "This clause is about {{}}.", "This provision concerns {{}}.", "The clause type is {{}}." and ensemble by averaging or taking max of entailment scores across templates for better accuracy.
   If all fail, return 'Unknown' for all predictions.
4. PREPROCESS TEXT: For each text in df['{text_column}']:
   - Apply light normalization: trim whitespace, normalize quotes/dashes, remove obvious headers/footers/page artifacts, optionally strip noisy leading numbering
   - DO NOT remove legally meaningful cues like "shall/must/may" and punctuation
   - If text is long (>400 tokens), chunk into ~250-400 token segments with small overlap, then aggregate per-label scores using max over chunks to avoid truncation losing key signal
5. CLASSIFY: For each processed text:
   - If text is empty/NaN or classifier is None: append 'Unknown'
   - If text was chunked: process each chunk, get scores per label, aggregate using max across chunks per label, then select label with highest aggregated score
   - Else: result = classifier(str(text), candidate_labels=candidate_labels)
   - predicted = result['labels'][0]
   - Append predicted to predictions list
6. ASSIGN: df['extracted_clause_type'] = predictions

CRITICAL: 
- candidate_labels must be exactly: [{', '.join([f'"{ct}"' for ct in unique_values])}]
- Use pipeline('zero-shot-classification', model=MODEL_NAME, hypothesis_template="This clause is about {{}}.")
- Look for DeBERTa-v3 based zero-shot classification models - these typically achieve best accuracy for this task
- Generic CLIP or vision-language models lead to poor accuracy - you need NLI-based zero-shot text classification models
- Extract result['labels'][0] from output
- For long clauses, chunk and aggregate scores using max to avoid truncation loss
- If many clause types, consider two-stage approach (coarse family → fine subtype)"""
    }

    nl_prompt = """Classify financial contract clauses from the contracts-clauses-datasets (mohammedalrashidan/contracts-clauses-datasets, https://www.kaggle.com/datasets/mohammedalrashidan/contracts-clauses-datasets) into their clause types. Use zero-shot text classification with NLI-based models via the pipeline function. 

Look for high-quality zero-shot classification models - prioritize DeBERTa-v3 based models specifically designed for zero-shot classification tasks, as these typically achieve best accuracy. Generic models lead to poor accuracy - you need NLI-based zero-shot text classification models trained on entailment datasets.

For best results: (1) Enhance labels with mini-definitions and synonyms as NLI models score entailment better with descriptive labels, (2) Test multiple hypothesis templates ("This clause is about {{}}.", "This provision concerns {{}}.", "The clause type is {{}}.") and ensemble scores, (3) For long clauses, chunk into ~250-400 token segments with overlap and aggregate per-label scores using max over chunks, (4) Apply light normalization (whitespace cleanup, normalize quotes/dashes, remove headers/footers) while keeping legal cues like "shall/must/may", (5) If many clause types, consider two-stage zero-shot pass (coarse family → fine subtype).

Extract candidate_labels directly from the clause types explicitly provided in the prompt. Create candidate_labels as a Python list of strings matching those exact clause types. If using AutoModel classes, the correct name is AutoModelForSeq2SeqLM (NOT AutoModelForSequence2SeqLM)."""

    with track_code_generation_cost() as costs:
        result_op = data_op.sem_extract_features(
            nl_prompt=nl_prompt,
            input_columns=[text_column],
            name="extract_clause_type",
            output_columns=output_columns,
            generate_via_code=True,
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
    print(f"  Code generation cost: ${total_cost:.6f}")

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

        df = df.sort_values(by="id").reset_index(drop=True)
        result_df = result_df.sort_values(by="id").reset_index(drop=True)

        # Debug prints and sum matches calculation
        if "clause_type" in df.columns and "extracted_clause_type" in result_df.columns:
            print(df[["id", "clause_type"]])
            print(result_df[["id", "extracted_clause_type"]])

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
    sempipes.update_config(llm_for_code_generation=sempipes.LLM(name="gemini/gemini-2.5-flash"))

    df = load_contracts_dataset("10000")
    print(f"\nLoaded dataset with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    text_column = "clause_text"
    num_repeats = 5
    try:
        all_results, accuracies, total_cost = extract_with_repeats_sempipes(
            df, text_column=text_column, num_repeats=num_repeats
        )

        result_df = all_results[-1]

        print(f"\nOriginal columns: {list(df.columns)}")
        print(f"New extracted columns: {[col for col in result_df.columns if col not in df.columns]}")

        average_accuracy = np.round(np.mean(accuracies), 3) if accuracies else 0.0
        average_accuracy_std = np.round(np.std(accuracies), 3) if accuracies else 0.0
        print(f"\nAverage accuracy across all repeats: {average_accuracy} +- {average_accuracy_std}")

        print("\nCOST SUMMARY (Code Generation)")
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
