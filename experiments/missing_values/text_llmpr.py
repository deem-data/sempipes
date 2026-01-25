#!/usr/bin/env python3
"""
Script for LLM-based batched data imputation using LiteLLM.

This script processes the restaurant dataset with missing values,
using LiteLLM for batched LLM calls. Model and prompt can be adjusted
via command line arguments or environment variables.
"""

import argparse
import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from litellm import batch_completion, APIConnectionError
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


def build_imputation_prompt(
    row_data: str, task: str = "restaurant"
) -> str:
    if task == "restaurant":
        prompt = f"""name: oceana. addr: 55 e. 54th st.. phone: 212/759-5941. type: seafood. What is the city? new york

name: oceana. addr: 55 e. 54th st.. phone: 212-759-5941. type: seafood. What is the city? new york city

name: matsuhisa. addr: 129 n. la cienega blvd.. phone: 310/659-9639. type: asian. What is the city? beverly hills

name: silver skillet, the, addr: 200 14th st. nw., phone: 404-874-1388, type: coffee shops. What is the city? atlanta

name: zab-e-lee, addr: 4837 old national hwy., phone: 404-768-2705, type: thai. What is the city? college park

name: bill's place, addr: 2315 clement st., phone: 415-221-5262, type: hamburgers. What is the city? san francisco

name: cafe '50s, addr: 838 lincoln blvd., phone: 310-399-1955, type: american. What is the city? venice

name: aureole, addr: 34 e. 61st st., phone: 212/ 319-1660, type: american. What is the city? new york

name: cafe blanc, addr: 9777 little santa monica blvd., phone: 310-888-0108, type: pacific new wave. What is the city? beverly hills

name: cassell's, addr: 3266 w. sixth st., phone: 213-480-8668, type: hamburgers. What is the city? la

name: diaghilev, addr: 1020 n. san vicente blvd., phone: 310-854-1111, type: russian. What is the city? w. hollywood

name: grill the, addr: 9560 dayton way, phone: 310-276-0615, type: american (traditional). What is the city? beverly hills

name: aquavit, addr: 13 w. 54th st., phone: 212/307-7311, type: continental. What is the city? new york

name: adriano's ristorante, addr: 2930 beverly glen circle, phone: 310/475-9807, type: italian. What is the city? los angeles

{row_data}. What is the city? Answer only city name."""

    else:
        prompt = f"""name: Transcend 8GB Compact Flash Card (133x) - TS8GCF133. description: Transcend 8GB CompactFlash Card (133x). Who is the manufacturer? TRANSCEND INFORMATION

name: LG 42LG30 42' LCD TV. description: LG 42LG30 42' LCD HDTV - 12,000:1 Dynamic Contrast Ratio - Invisible Speaker. Who is the manufacturer? LG Electronics

name: Speck Products SeeThru Case for Apple MacBook Air - MBA-PNK-SEE. description: Plastic - Pink. Who is the manufacturer? Speck Products
          
name: Peerless Universal Tilt Wall Mount. description: Peerless Smart Mount ST660P Universal Tilt Wall Mount for 37' to 60' Screens (Black) up to 200lbs. Who is the manufacturer? Peerless

name: Apple Time Capsule Network Hard Drive - MB277LL/A. description: 1TB - Type A USB. Who is the manufacturer? Apple

name: Sirius SUPH1 Sirius Universal Home Kit. description: Sirius Satellite Radio Receiver. Who is the manufacturer? Sirius

name: OmniMount TV Top Shelf Mount. description: OmniMount CCH1B Set-Top Center-Channel Shelf. Who is the manufacturer? OMNIMOUNT SYSTEMS, INC

name: Monster Cable iFreePlay Cordless Headphone - AI SH HPHONE. description: Connectivity: Wireless - Stereo - Behind-the-neck. Who is the manufacturer? Monster

name: Pure Digital Flip Mino Digital Camcorder - F360B. description: Flip Video Mino 60 min Black. Who is the manufacturer? Pure Digital Technol

name: Elgato EyeTV Hybrid Analog/Digital TV Tuner Stick - 10020630. description: Elgato EyeTV Hybrid TV Tuner Stick for Analog and HDTV Reception - USB. Who is the manufacturer? ELGATO SYSTEMS                               

{row_data}. Who is the manufacturer? Answer ONLY MANUFACTURER NAME, without additional text or formatting. Look at the name of the product and the description to infer the manufacturer name."""
    return prompt.strip()


def build_messages(prompt: str, task: str = "restaurant") -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Impute the city name from the data provided restaurant details" if task == "restaurant" else "Impute the manufacturer name from the data provided product details"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


def batch_impute_with_llm(
    df: pd.DataFrame,
    target_column: str,
    model: str,
    batch_size: int = 20,
    temperature: float = 0.0,
    **llm_kwargs
) -> pd.DataFrame:
    df_result = df.copy()

    if target_column not in df_result.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in input DataFrame. "
            f"Available columns: {list(df_result.columns)}"
        )
    
    # Find rows with missing values
    missing_mask = df_result[target_column].isna()
    rows_to_impute = df_result[missing_mask]
    
    if len(rows_to_impute) == 0:
        print(f"\t> No missing values found in column '{target_column}'.")
        return df_result
    
    print(f"\t> Found {len(rows_to_impute)} rows with missing values in '{target_column}'.")
    
    # Build prompts for all rows to impute
    print(f"\t> Building prompts for {len(rows_to_impute)} rows...")
    message_batches = []
    for _, row in rows_to_impute.iterrows():
        if target_column == "city":
            # row_data = f"{row["type"]} restaurant {row["name"]} located at {row["addr"]}, phone number is {row["phone"]}."
            row_data = "; ".join([f"{col}: {row[col]}" for col in df.columns if col != target_column])
            message_batches.append(build_messages(build_imputation_prompt(row_data, task="restaurant")))
        else:
            row_data = ". ".join([f"{col}: {row[col]}" for col in df.columns if col != target_column])
            message_batches.append(build_messages(build_imputation_prompt(row_data, task="buy"), task="buy"))
    
    # Process in batches
    print(f"\t> Querying '{model}' with {len(message_batches)} requests in batches of size {batch_size}...")
    
    all_results = []
    llm_params = {"temperature": temperature, **llm_kwargs} if "openai/gpt-5" not in model else {**llm_kwargs}
    
    for start_idx in tqdm(range(0, len(message_batches), batch_size), desc="Processing batches"):
        batch = message_batches[start_idx:start_idx + batch_size]
        
        try:
            responses = batch_completion(
                model=model,
                messages=batch,
                **llm_params
            )
            
            for i, response in enumerate(responses):
                try:
                    if isinstance(response, Exception):
                        print(f"\n> Error in response {i}: {type(response).__name__}: {response}")
                        all_results.append(None)
                    elif hasattr(response, 'choices') and len(response.choices) > 0:
                        message = response.choices[0].message
                        # Handle both dict and object formats
                        if isinstance(message, dict):
                            content = message.get("content", "")
                        elif hasattr(message, "content"):
                            content = message.content
                        else:
                            print(f"\n> Unexpected message format in response {i}: {type(message)}")
                            all_results.append(None)
                            continue
                        # Strip whitespace from result
                        all_results.append(content.strip() if content else None)
                    else:
                        print(f"\n> Unexpected response format: {type(response)}")
                        all_results.append(None)
                except Exception as e:
                    print(f"\n> Error processing response {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append(None)
                    
        except Exception as e:
            print(f"\n> Batch completion failed: {e}")
            all_results.extend([None] * len(batch))
    
    # Assign results back to DataFrame
    imputed_indices = df_result[missing_mask].index
    for idx, result in zip(imputed_indices, all_results):
        if result is not None:
            df_result.loc[idx, target_column] = result
        else:
            print(f"\t> Warning: Failed to impute row {idx}")
    
    print(f"\t> Successfully imputed {sum(1 for r in all_results if r is not None)} values.")
    
    return df_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash"),
        help="LiteLLM model name (e.g., 'openai/gpt-4', 'gemini/gemini-2.5-flash')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of requests per batch"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the model"
    )
    parser.add_argument(
        "--nreps",
        type=int,
        default=5,
        help="Number of repeated runs for imputation"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="city", #"manufacturer",
        help="Column name to impute"
    )    
    args = parser.parse_args()
    
    data_dir = "experiments/missing_values/textual_data/Restaurant" if args.target_column == "city" else "experiments/missing_values/textual_data/Buy"
    
    # Load restaurant dataset
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    valid_df = pd.read_csv(f"{data_dir}/valid.csv")
    test_df_clean = pd.read_csv(f"{data_dir}/test.csv")

    # Infer target column if not provided, and validate it exists for evaluation.
    # (We later compare imputed values against the original test column.)
    if args.target_column is None:
        if "city" in test_df_clean.columns:
            args.target_column = "city"
        elif "manufacturer" in test_df_clean.columns:
            args.target_column = "manufacturer"
        else:
            raise ValueError(
                "Could not infer --target-column from test.csv. "
                f"Available columns: {list(test_df_clean.columns)}. "
                "Please pass --target-column explicitly."
            )
    elif args.target_column not in test_df_clean.columns:
        raise ValueError(
            f"--target-column '{args.target_column}' not found in test.csv. "
            f"Available columns: {list(test_df_clean.columns)}. "
            "Pass the correct --target-column for this dataset (e.g. 'manufacturer' for Buy)."
        )
    
    print(f"Loaded {len(train_df)} train, {len(valid_df)} valid, {len(test_df_clean)} test rows.")
    print(f"\nImputing missing values in '{args.target_column}' column...")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of repeats: {args.nreps}")

    f1_scores = []
    accuracy_scores = []

    for repeat in range(args.nreps):
        print(f"\n=== Repeat {repeat + 1}/{args.nreps} ===")

        # Create test dataset with missing values
        test_df = test_df_clean.copy(deep=True)
        test_df[args.target_column] = None

        # Combine all data for context (but only impute test rows)
        # combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        combined_df = test_df

        # Perform imputation
        imputed_df = batch_impute_with_llm(
            df=combined_df,
            target_column=args.target_column,
            model=args.model,
            batch_size=args.batch_size,
            temperature=args.temperature,
        )
        
        # Extract test results
        imputed_test = imputed_df.copy()
        imputed_test_city = imputed_test[args.target_column].str.lower()
        test_df_clean[args.target_column] = test_df_clean[args.target_column].str.lower()
        
        # Print some statistics
        print(f"\nImputation statistics (run {repeat + 1}):")
        print(f"  Total test rows: {len(test_df_clean)}")
        print(f"  Successfully imputed: {imputed_test_city.notna().sum()}")
        print(f"  Failed imputations: {imputed_test_city.isna().sum()}")
        
        # Show sample results only for the first run to avoid excessive output
        if repeat == 0:
            print(f"\nSample imputed values (run {repeat + 1}):")
            sample_size = min(10, len(imputed_test_city))
            for i in range(sample_size):
                original = test_df_clean[args.target_column].iloc[i]
                imputed = imputed_test_city.iloc[i]
                if imputed != original:
                    print(f"  Row {test_df_clean.iloc[i]}: Original='{original}' -> Imputed='{imputed}'")
        
        # Calculate F1 score and accuracy
        f1 = f1_score(test_df_clean[args.target_column], imputed_test_city, average="macro")
        accuracy = accuracy_score(test_df_clean[args.target_column], imputed_test_city)
        print(f"F1 score (run {repeat + 1}): {f1}")
        print(f"Accuracy (run {repeat + 1}): {accuracy}")

        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    if args.nreps > 1:
        print("\n=== Summary over repeats ===")
        print(f"F1 score: mean={np.mean(f1_scores):.4f}, std={np.std(f1_scores):.4f}")
        print(f"Accuracy: mean={np.mean(accuracy_scores):.4f}, std={np.std(accuracy_scores):.4f}")

if __name__ == "__main__":
    main()

