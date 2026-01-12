import os

# Set SCIPY_ARRAY_API=1 to enable scipy array API compatibility. Must be set before any scipy/sklearn imports.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import dotenv

dotenv.load_dotenv()

from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import skrub  # noqa: E402
from litellm import completion, completion_cost  # noqa: E402

import sempipes  # noqa: E402


def load_chestxray_dataset(dataset_size: str = "1000"):
    """Load the Chest X-Ray Pneumonia dataset."""
    data_path = f"tests/data/chestxray-dataset-{dataset_size}/chestxray.csv"
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


def extract_class_with_sempipes(df: pd.DataFrame, image_column: str = "path") -> tuple[pd.DataFrame, float]:
    """
    Extract class in a single sem_extract_features call using sempipes.

    Args:
        df: DataFrame containing the dataset
        image_column: Name of the column containing image paths

    Returns:
        Tuple of (DataFrame with extracted_class column added, total cost in USD for code generation)
    """
    valid_mask = df[image_column].notna() & (df[image_column].astype(str).str.strip() != "")
    df_filtered = df[valid_mask].copy()

    valid_image_mask = df_filtered[image_column].apply(lambda x: Path(str(x)).exists())
    df_minimal = df_filtered[valid_image_mask].copy()

    if len(df_minimal) == 0:
        result_df = df.copy()
        result_df["extracted_class"] = "Unknown"
        return result_df, 0.0

    df_minimal = df_minimal[[image_column]].copy().reset_index()
    original_index_name = df.index.name if df.index.name is not None else "index"

    data_op = skrub.var("chestxray_data", df_minimal)

    # Get unique class values for the prompt
    unique_values = (
        sorted([str(v) for v in df["class"].dropna().unique() if pd.notna(v)]) if "class" in df.columns else []
    )
    values_str = ", ".join(unique_values)
    values_hint = f" The possible values are: {values_str}." if unique_values else ""

    base_prompt = f"Classify chest X-ray images into NORMAL and PNEUMONIA classes. The `{image_column}` column contains the chest X-ray image path. \n\nCRITICAL: You MUST use a biomedical CLIP model trained on medical images. DO NOT use general-purpose CLIP models - they give poor accuracy (around 0.5) for medical images because they are not trained on medical data. \n\nMANDATORY MODEL SELECTION: \n1. YOU MUST search for and use a biomedical CLIP model. Look for models with identifiers containing 'BiomedCLIP', 'biomedical', 'medical', 'PubMed', or 'BioMed' in their names. These models are trained on medical images and will give much better accuracy. \n2. If you cannot find a biomedical CLIP model, look for a model specifically trained on chest X-ray images. If it uses TensorFlow/Keras, use tf.saved_model.load() or tf.keras.models.load_model() - do NOT use transformers.pipeline() for TensorFlow models. \n3. ONLY as absolute last resort if no biomedical models exist, you may use a general CLIP model, but this is strongly discouraged and will give poor accuracy. \n\nREQUIRED IMPLEMENTATION FOR BIOMEDICAL CLIP: \n- Import: from open_clip import create_model_from_pretrained, get_tokenizer. Also import: from PIL import Image, import torch. \n- Load model: model, preprocess = create_model_from_pretrained('hf-hub:MODEL_IDENTIFIER'), tokenizer = get_tokenizer('hf-hub:MODEL_IDENTIFIER') where MODEL_IDENTIFIER is a biomedical CLIP model. \n- Set device: device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'), model.to(device), model.eval(). \n- Use descriptive labels: labels = ['a normal chest X-ray with clear lung fields and no opacities', 'a chest X-ray showing pneumonia with visible lung opacities and infiltrates'], template = 'this is a photo of '. \n- Create label mapping: label_map = {{'a normal chest X-ray with clear lung fields and no opacities': 'NORMAL', 'a chest X-ray showing pneumonia with visible lung opacities and infiltrates': 'PNEUMONIA'}}. \n- Process each image: Load image = Image.open(image_path), preprocess: image_tensor = preprocess(image).unsqueeze(0).to(device), tokenize: texts = tokenizer([template + l for l in labels], context_length=256).to(device). \n- Get predictions: with torch.no_grad(): image_features, text_features, logit_scale = model(image_tensor, texts), logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1), sorted_indices = torch.argsort(logits, dim=-1, descending=True), predicted_label = labels[sorted_indices[0][0]], map to 'NORMAL' or 'PNEUMONIA' using label_map. \n- Error handling: Wrap in try-except with ImportError or Exception, return 'Unknown' on error. \n- DO NOT import warnings, sys, os, or other blocked modules. \n"

    feature_prompt = f"""Classify the chest X-ray image to determine if it shows pneumonia or is normal.
The class should be one of: {values_hint}, or Unknown if not determinable.
Return only the class name."""

    output_columns = {"extracted_class": base_prompt + feature_prompt}

    nl_prompt = "You are a helpful medical imaging expert specializing in chest X-ray analysis. Classify chest X-ray images to detect pneumonia. \n\nCRITICAL: You MUST use a biomedical CLIP model trained on medical images. DO NOT use general-purpose CLIP models - they give poor accuracy (around 0.5) for medical images because they are not trained on medical data. \n\nMANDATORY: Search for and use a biomedical CLIP model with identifiers containing 'BiomedCLIP', 'biomedical', 'medical', 'PubMed', or 'BioMed'. These models are specifically trained on medical images and will give much better accuracy. \n\nREQUIRED IMPLEMENTATION: \n1. Import: from open_clip import create_model_from_pretrained, get_tokenizer. Also import: from PIL import Image, import torch. \n2. Load biomedical CLIP model: model, preprocess = create_model_from_pretrained('hf-hub:MODEL_IDENTIFIER'), tokenizer = get_tokenizer('hf-hub:MODEL_IDENTIFIER') where MODEL_IDENTIFIER is a biomedical CLIP model. \n3. Set device: device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'), model.to(device), model.eval(). \n4. Use descriptive labels: labels = ['a normal chest X-ray with clear lung fields and no opacities', 'a chest X-ray showing pneumonia with visible lung opacities and infiltrates'], template = 'this is a photo of '. \n5. Create mapping: label_map = {'a normal chest X-ray with clear lung fields and no opacities': 'NORMAL', 'a chest X-ray showing pneumonia with visible lung opacities and infiltrates': 'PNEUMONIA'}. \n6. Process each image ONE AT A TIME: Load image = Image.open(image_path), preprocess: image_tensor = preprocess(image).unsqueeze(0).to(device), tokenize: texts = tokenizer([template + l for l in labels], context_length=256).to(device). \n7. Get prediction: with torch.no_grad(): image_features, text_features, logit_scale = model(image_tensor, texts), logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1), sorted_indices = torch.argsort(logits, dim=-1, descending=True), predicted_label = labels[sorted_indices[0][0]], map to 'NORMAL' or 'PNEUMONIA' using label_map. \n8. Error handling: Wrap in try-except with ImportError or Exception, return 'Unknown' on error. \n\nDO NOT use transformers.pipeline() with general CLIP models. Use open_clip library with biomedical CLIP models instead. The images are chest X-ray radiographs showing lung conditions."

    with track_code_generation_cost() as costs:
        result_op = data_op.sem_extract_features(
            nl_prompt=nl_prompt,
            input_columns=[image_column],
            name="extract_class",
            output_columns=output_columns,
            generate_via_code=True,
        )
        result_df_minimal = result_op.skb.eval()

    total_cost = sum(costs) if costs else 0.0
    result_df = df.copy().reset_index()

    extracted_col_name = "extracted_class"
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

    print("\nExtracted class from images in a single sem_extract_features call")
    print(f"  Code generation cost: ${total_cost:.6f}")

    if "class" in df.columns and extracted_col_name in result_df.columns:
        accuracy = compute_accuracy(df["class"], result_df[extracted_col_name])
        print(f"  Accuracy for class: {accuracy:.3f} ({accuracy*100:.2f}%)")

    return result_df, total_cost


def extract_with_repeats_sempipes(
    df: pd.DataFrame, image_column: str = "path", num_repeats: int = 3
) -> tuple[list[pd.DataFrame], list[float], float]:
    """
    Extract class multiple times using sempipes and compute accuracy statistics.

    Args:
        df: DataFrame containing the dataset
        image_column: Name of the column containing image paths
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

        result_df, repeat_cost = extract_class_with_sempipes(df, image_column)

        total_cost += repeat_cost

        df = df.sort_values(by="filename").reset_index(drop=True)
        result_df = result_df.sort_values(by="filename").reset_index(drop=True)

        # Debug prints and sum matches calculation
        if "class" in df.columns and "extracted_class" in result_df.columns:
            print(df[["filename", "class"]])
            print(result_df[["filename", "extracted_class"]])

            sum_matches = sum(
                df["class"].astype(str).str.strip().str.lower()
                == result_df["extracted_class"].astype(str).str.strip().str.lower()
            )
            print(f"Sum of matches: {sum_matches}, accuracy {sum_matches / len(df):.3f}")

        accuracy = compute_accuracy(df["class"], result_df["extracted_class"])
        accuracies.append(accuracy)
        print(f"\n  Accuracy (repeat {repeat_num + 1}): {accuracy:.3f} ({accuracy*100:.2f}%)")

        all_results.append(result_df)

    return all_results, accuracies, total_cost


if __name__ == "__main__":
    sempipes.update_config(batch_size_for_batch_processing=10)
    sempipes.update_config(llm_for_code_generation=sempipes.LLM(name="gemini/gemini-2.5-flash"))

    df = load_chestxray_dataset("1000")
    print(f"\nLoaded dataset with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    num_repeats = 5
    try:
        all_results, accuracies, total_cost = extract_with_repeats_sempipes(
            df, image_column="path", num_repeats=num_repeats
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

        output_path = "experiments/feature_extraction/extracted_features_sempipes_chestxray.csv"
        result_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Feature extraction complete!")
