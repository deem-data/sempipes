#!/usr/bin/env python3

import os
import warnings
import resource
from typing import List

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import accuracy_score, f1_score
import sempipes


warnings.filterwarnings("ignore")
np.random.seed(0)

# This appears to be necessary for not running into "too many open files" errors.
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))


def experiment_sempipes(
    prompt_unconstrained: str,
    prompt_constrained: str,
    nreps: int = 3,
    with_llm_only: bool = True,
    data_dir: str = "experiments/missing_values/textual_data/Restaurant",
    target_column: str = "city",
    dataset_name: str = "restaurants",
) -> None:
    """Run a SemPipes imputation experiment for a generic dataset."""
    f1_scores: List[float] = []
    accuracy_scores: List[float] = []
    for repeat in range(nreps):
        df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
        df_valid = pd.read_csv(os.path.join(data_dir, "valid.csv"))
        df_test_clean = pd.read_csv(
            os.path.join(data_dir, "test.csv")
        )

        df_test = df_test_clean.copy(deep=True)

        # Mask out the target column in the test split
        df_test[target_column] = None

        df = pd.concat([df_train, df_valid, df_test])

        df[target_column] = df[target_column].apply(
            lambda x: x.lower() if isinstance(x, str) else x
        )

        dataset_ref = skrub.var(dataset_name, df)

        if with_llm_only:
            prompt = prompt_unconstrained
            impute_with_existing_values_only = False

        else:
            prompt = prompt_constrained
            impute_with_existing_values_only = True
            
        df_imputed = dataset_ref.sem_fillna(
            target_column=target_column,
            nl_prompt=prompt,
            impute_with_existing_values_only=impute_with_existing_values_only,
            with_llm_only=with_llm_only,
        ).skb.eval()

        df_imputed[target_column] = df_imputed[target_column].apply(
            lambda x: x.lower() if isinstance(x, str) else x
        )

        df_test_clean[target_column] = df_test_clean[target_column].apply(
            lambda x: x.lower() if isinstance(x, str) else x
        )

        # Slice only test rows from the imputed results
        test_start_idx = len(df_train) + len(df_valid)
        imputed_test = df_imputed[target_column].iloc[
            test_start_idx:
        ]

        f1 = f1_score(
            df_test_clean[target_column],
            imputed_test,
            average="macro",
        )
        acc = accuracy_score(
            df_test_clean[target_column],
            imputed_test,
        )

        for imputed, clean in zip(imputed_test, df_test_clean[target_column]):
            if imputed != clean:
                print(f"Imputed: {imputed}, Clean: {clean}")

        print(f"F1 score: {f1}")
        print(f"Accuracy: {acc}")
        f1_scores.append(f1)
        accuracy_scores.append(acc)

    print(
        f"Average F1 score: {np.mean(f1_scores)}, "
        f"Standard deviation: {np.std(f1_scores)}"
    )
    print(
        f"Average Accuracy score: {np.mean(accuracy_scores)}, "
        f"Standard deviation: {np.std(accuracy_scores)}"
    )


def main(
    model: str,
    nreps: int,
    data_dir: str,
    target_column: str,
    with_llm_only: bool,
) -> None:
    """Configure SemPipes LLMs and run the experiment."""
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name=model,
            parameters={"temperature": 1.0},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name=model,
            parameters={"temperature": 1.0},
        ),
    )
    print("Running LLM processing")
    print(f"Model: {model}")
    print(f"Number of repeats: {nreps}")
    print(f"Data directory: {data_dir}")
    print(f"With LLM only: {with_llm_only}")

    # Dataset-specific prompts passed into a single generic experiment function.
    if data_dir.endswith("Restaurant") and target_column == "city":
        restaurant_prompt_unconstrained = """Infer the city where the restaurant is located from relevant related attributes, especially the restaurant name, telephone number, and address. Column name is restaurant name, `addr` is address in USA, `phone` is telephone number, `type` is type of the food served at the restaurant.

Examples:
Seafood restaurant called 'Oceana' located at 55 e. 54th st., phone number is 212/759-5941. What is the city? New York

Seafood restaurant called 'Oceana' located at 55 e. 54th st., phone number is 212-759-5941. What is the city? New York City

Asian restaurant called 'Matsuhisa' located at 129 n. la cienega blvd., phone number is 310/659-9639. What is the city? Beverly Hills

Coffee shop called 'Silver Skillet' located at 200 14th st. nw., phone number is 404-874-1388. What is the city? Atlanta

Thai restaurant called 'Zab-e-Lee' located at 4837 old national hwy., phone number is 404-768-2705. What is the city? College Park

Hamburgers restaurant called Bill's Place located at 2315 clement st., phone number is 415-221-5262. What is the city? San Francisco

American cafe called '50s located at 838 lincoln blvd., phone number is 310-399-1955. What is the city? Venice

Russian restaurant called Diaghilev located at 1020 n. san vicente blvd., phone number is 310-854-1111. What is the city? W. Hollywood

Italian restaurant called Adriano's Ristorante located at 2930 beverly glen circle, phone number is 310/475-9807. What is the city? Los Angeles

What is the city?"""

        restaurant_prompt_constrained = (
            "Infer the city where the restaurant is located from relevant "
            "related attributes, especially the restaurant name and "
            "address. Keep the same style of the city name as in the "
            "original dataset. Please pay attention to the restaurant "
            "name and phone number since city in only name can be "
            "misleading."
        )

        experiment_sempipes(
            nreps=nreps,
            with_llm_only=with_llm_only,
            data_dir=data_dir,
            target_column="city",
            dataset_name="restaurants",
            prompt_unconstrained=restaurant_prompt_unconstrained,
            prompt_constrained=restaurant_prompt_constrained,
        )

    else:
#         buy_prompt_unconstrained = """Infer the manufacturer name fraom the product name and description. Column `name` is product name, `description` is description of the product.

# Examples:
# name: Transcend 8GB Compact Flash Card (133x) - TS8GCF133. description: Transcend 8GB CompactFlash Card (133x). Who is the manufacturer? TRANSCEND INFORMATION

# name: LG 42LG30 42' LCD TV. description: LG 42LG30 42' LCD HDTV - 12,000:1 Dynamic Contrast Ratio - Invisible Speaker. Who is the manufacturer? LG Electronics

# name: Speck Products SeeThru Case for Apple MacBook Air - MBA-PNK-SEE. description: Plastic - Pink. Who is the manufacturer? Speck Products
          
# name: Peerless Universal Tilt Wall Mount. description: Peerless Smart Mount ST660P Universal Tilt Wall Mount for 37' to 60' Screens (Black) up to 200lbs. Who is the manufacturer? Peerless

# name: Apple Time Capsule Network Hard Drive - MB277LL/A. description: 1TB - Type A USB. Who is the manufacturer? Apple

# name: Sirius SUPH1 Sirius Universal Home Kit. description: Sirius Satellite Radio Receiver. Who is the manufacturer? Sirius

# name: OmniMount TV Top Shelf Mount. description: OmniMount CCH1B Set-Top Center-Channel Shelf. Who is the manufacturer? OMNIMOUNT SYSTEMS, INC

# name: Monster Cable iFreePlay Cordless Headphone - AI SH HPHONE. description: Connectivity: Wireless - Stereo - Behind-the-neck. Who is the manufacturer? Monster

# name: Pure Digital Flip Mino Digital Camcorder - F360B. description: Flip Video Mino 60 min Black. Who is the manufacturer? Pure Digital Technology

# name: Elgato EyeTV Hybrid Analog/Digital TV Tuner Stick - 10020630. description: Elgato EyeTV Hybrid TV Tuner Stick for Analog and HDTV Reception - USB. Who is the manufacturer? ELGATO SYSTEMS                               

# Who is the manufacturer? Answer ONLY MANUFACTURER NAME."""

        buy_prompt_unconstrained = """Infer the manufacturer name from the product name and description. Column `name` is product name, `description` is description of the product.

Examples:
Product is named 'Transcend 8GB Compact Flash Card (133x) - TS8GCF133', and its description is 'Transcend 8GB CompactFlash Card (133x)'. Who is the manufacturer? transcend information

Product is named 'LG 42LG30 42' LCD TV', and its description is 'LG 42LG30 42' LCD HDTV - 12,000:1 Dynamic Contrast Ratio - Invisible Speaker'. Who is the manufacturer? lg electronics

Product is named 'Speck Products SeeThru Case for Apple MacBook Air - MBA-PNK-SEE', and its description is 'Plastic - Pink'. Who is the manufacturer? speck products
          
Product is named 'Peerless Universal Tilt Wall Mount', and its description is 'Peerless Smart Mount ST660P Universal Tilt Wall Mount for 37' to 60' Screens (Black) up to 200lbs'. Who is the manufacturer? peerless

Product is named 'Apple Time Capsule Network Hard Drive - MB277LL/A', and its description is '1TB - Type A USB'. Who is the manufacturer? apple

Product is named 'Sirius SUPH1 Sirius Universal Home Kit', and its description is 'Sirius Satellite Radio Receiver'. Who is the manufacturer? sirius

Product is named 'OmniMount TV Top Shelf Mount', and its description is 'OmniMount CCH1B Set-Top Center-Channel Shelf'. Who is the manufacturer? omnimount systems, inc

Product is named 'Monster Cable iFreePlay Cordless Headphone - AI SH HPHONE', and its description is 'Connectivity: Wireless - Stereo - Behind-the-neck'. Who is the manufacturer? monster

Product is named 'Pure Digital Flip Mino Digital Camcorder - F360B', and its description is 'Flip Video Mino 60 min Black'. Who is the manufacturer? pure digital technol

Product is named 'Elgato EyeTV Hybrid Analog/Digital TV Tuner Stick - 10020630', and its description is 'Elgato EyeTV Hybrid TV Tuner Stick for Analog and HDTV Reception - USB'. Who is the manufacturer? elgato systems                               

Who is the manufacturer? Answer ONLY MANUFACTURER NAME, Look at the existing names of the products to infer the manufacturer name."""

        buy_prompt_constrained = (
            "Infer the manufacturer name from the product name and description"
        )
        
        experiment_sempipes(
            nreps=nreps,
            with_llm_only=with_llm_only,
            data_dir=data_dir,
            target_column=target_column,
            dataset_name="buy",
            prompt_unconstrained=buy_prompt_unconstrained,
            prompt_constrained=buy_prompt_constrained,
        )


if __name__ == "__main__":
    # Restaurant experiment\
    model = "gemini/gemini-2.5-flash"
    main(
        model=model,
        nreps=8,
        data_dir="experiments/missing_values/textual_data/Restaurant",
        target_column="city",
        with_llm_only=True,
    )
    
    # # Buy experiment LLM
    # main(
    #     model=model,
    #     nreps=5,
    #     data_dir="experiments/missing_values/textual_data/Buy",
    #     target_column="manufacturer",
    #     with_llm_only=True,
    # )