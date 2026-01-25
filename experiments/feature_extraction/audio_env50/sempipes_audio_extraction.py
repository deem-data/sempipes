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
def track_code_generation_cost():
    costs = []
    call_count = [0]

    from litellm import completion as original_completion

    import sempipes.llm.llm as llm_module

    def tracked_completion(*args, **kwargs):
        call_count[0] += 1
        response = original_completion(*args, **kwargs)
        try:
            cost = completion_cost(completion_response=response)
            if cost is None:
                print(f"Warning: completion_cost returned None for call #{call_count[0]}")
                cost = 0.0
            elif isinstance(cost, dict):
                print(f"Warning: completion_cost returned dict {cost} for call #{call_count[0]}")
                cost = cost.get("total_cost", cost.get("cost", 0.0))
            elif not isinstance(cost, (int, float)):
                print(
                    f"Warning: completion_cost returned unexpected type {type(cost)}: {cost} for call #{call_count[0]}"
                )
                cost = 0.0

            costs.append(float(cost))
            print(f"  [Cost tracking] Call #{call_count[0]}: ${float(cost):.6f}")
        except Exception as e:
            print(f"Warning: Could not calculate cost for call #{call_count[0]}: {e}")
            costs.append(0.0)
        return response

    original_module_completion = llm_module.completion
    llm_module.completion = tracked_completion

    try:
        yield costs
    finally:
        llm_module.completion = original_module_completion
        print(f"  [Cost tracking] Total completion calls: {call_count[0]}, Total cost: ${sum(costs):.6f}")


def extract_category_with_sempipes(df: pd.DataFrame, audio_column: str = "filename") -> tuple[pd.DataFrame, float]:
    """
    Extract category in a single sem_extract_features call using sempipes.

    Args:
        df: DataFrame containing the sounds dataset
        audio_column: Name of the column containing audio file paths

    Returns:
        Tuple of (DataFrame with extracted_category column added, total cost in USD for code generation)
    """
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
    candidate_labels = unique_values

    candidate_labels_desc = ""
    if candidate_labels:
        labels_list = ", ".join([f'"{label}"' for label in candidate_labels])
        candidate_labels_desc = f"The candidate labels are: [{labels_list}]."

    nl_prompt = "You are a helpful audio environmental sound classification expert. Extract the category from the audio file. Return only the value, no additional text."

    output_columns = {
        "extracted_category": f"""Extract the category (e.g., animals, water, human speech, exterior/urban sounds, natural soundscapes) from the environmental sound audio files. Return only the category name, no additional text. Use 'Unknown' if not determinable.

CRITICAL: You MUST follow these exact steps to generate working code. Do NOT skip any step.

STEP 1 - IMPORTS (REQUIRED):
Import soundfile, transformers, and numpy (as np) at the top of your code. 
DO NOT import torchaudio - it requires dependencies that are NOT installed and will cause runtime errors.

STEP 2 - INITIALIZE CLASSIFIER (REQUIRED):
Create a zero-shot audio classification pipeline using the transformers library. 
Initialize a variable called classifier to None, then wrap the pipeline creation in a try-except block. 
Inside the try block, create a pipeline with task="zero-shot-audio-classification". 
For the model, use a contrastive language-audio pretraining (CLAP) model from a large-scale audio-text dataset project. 
Look for a model that uses HTSAT architecture in an unfused variant, trained on a large collection of audio-text pairs (over 600K samples). 
The model should be from an organization that focuses on open datasets and multimodal learning. 
If this model fails, try an alternative CLAP model with the same task. If both fail, leave classifier as None. 

STEP 3 - SET UP CANDIDATE LABELS (REQUIRED):
{candidate_labels_desc}
Create a list variable called candidate_labels containing these labels as strings (properly quoted).

STEP 4 - PROCESS AUDIO FILES (REQUIRED):
Create an empty list called predictions. Iterate through each audio file path in the dataframe column named '{audio_column}'. 
Wrap each iteration in a try-except block to handle errors gracefully.

Inside the try block:
- First check if classifier is None. If it is, append 'Unknown' to predictions and continue to the next file.
- Load the audio file using soundfile.read(audio_path), which returns (audio_data, sample_rate).
- Convert audio_data to a numpy array with dtype=np.float32.
- If audio_data has more than one dimension (stereo/multi-channel), convert to mono by taking the mean across channels (axis 1).
- Normalize audio_data to range [-1, 1] by dividing by the maximum absolute value (if > 0).
- Pass the normalized numpy array directly to classifier(audio_data, candidate_labels=candidate_labels). 
  DO NOT pass a dictionary - the classifier expects a numpy array directly.
- The classifier returns a list of dictionaries sorted by score (descending), each with 'label' and 'score' keys.
- Extract results[0]['label'] (highest confidence) and append to predictions. If results is empty/None, append 'Unknown'.

In the except block, print an error message with audio_path and exception details, then append 'Unknown' to predictions.

STEP 5 - CREATE OUTPUT COLUMN (REQUIRED):
Assign the predictions list to a new column named exactly 'extracted_category' in the dataframe.
"""
    }

    with track_code_generation_cost() as costs:
        result_op = data_op.sem_extract_features(
            nl_prompt=nl_prompt,
            input_columns=[audio_column],
            name="extract_category",
            output_columns=output_columns,
            generate_via_code=True,
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

        if original_index_name in result_df.columns:
            result_df = result_df.set_index(original_index_name)
            result_df.index.name = df.index.name

    print("\nExtracted category from audio files using sempipes (transformers models)...")
    print(f"  Cost for category: ${total_cost:.6f}")

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

        print(f"\n{'='*80}")
        print(f"COST FOR REPEAT {repeat_num + 1} (Code Generation):")
        print(f"{'='*80}")
        print(f"  category: ${repeat_cost:.6f}")
        print(f"  Total: ${repeat_cost:.6f}")
        print(f"{'='*80}\n")

        # Sort by filename before computing accuracy
        df = df.sort_values(by="filename").reset_index(drop=True)
        result_df = result_df.sort_values(by="filename").reset_index(drop=True)

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
            print(f"  Accuracy for category: {accuracy:.3f} ({accuracy*100:.2f}%)")

        all_results.append(result_df)

    return all_results, accuracies, total_cost


if __name__ == "__main__":
    sempipes.update_config(batch_size_for_batch_processing=10)
    sempipes.update_config(llm_for_code_generation=sempipes.LLM(name="gemini/gemini-2.5-flash"))

    df = load_sounds_dataset("1000")
    print(f"\nLoaded dataset with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    num_repeats = 5

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

        print(f"\n{'='*80}")
        print("FINAL COST SUMMARY (Code Generation - All Repeats)")
        print(f"{'='*80}")
        print(f"  category: ${total_cost:.6f}")
        print(f"\n  Total cost for code generation (all repeats): ${total_cost:.6f}")

        output_path = "experiments/feature_extraction/extracted_features_sempipes_audio.csv"
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()
