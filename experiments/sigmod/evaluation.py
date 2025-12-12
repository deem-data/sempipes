import argparse
import os

import numpy as np
import pandas as pd
import sklearn.metrics


def __is_dataset_lexicographically_sorted(df, left_column_name, right_column_name):
    number_not_sorted_rows = len(df[df[left_column_name] > df[right_column_name]])
    return number_not_sorted_rows == 0


def __sort_dataset_lexicographically(df, left_column_name, right_column_name):
    for index, row in df[df[left_column_name] > df[right_column_name]].iterrows():
        left_instance_id = row[left_column_name]
        right_instance_id = row[right_column_name]
        df.at[index, left_column_name] = right_instance_id
        df.at[index, right_column_name] = left_instance_id


def __drop_dataset_duplicates(df):
    df.drop_duplicates(inplace=True)


def __get_dataset(dataset_path, left_column_name, right_column_name):
    df = pd.read_csv(dataset_path, skipinitialspace=True)
    df = df.rename(str.strip, axis="columns")
    if not __is_dataset_lexicographically_sorted(df, left_column_name, right_column_name):
        __sort_dataset_lexicographically(df, left_column_name, right_column_name)
    return df


def get_evaluation_dataset_with_predicted_label(
    evaluation_dataset_path,
    submission_dataset_path,
    dataset_id,
):
    evaluation_dataset = __get_dataset(evaluation_dataset_path, "lid", "rid")
    submission_dataset = __get_dataset(submission_dataset_path, "left_instance_id", "right_instance_id")

    assert len(submission_dataset) == 3000000

    if dataset_id == 1:
        submission_dataset = submission_dataset[:1000000]
        assert len(submission_dataset) == 1000000
    elif dataset_id == 2:
        submission_dataset = submission_dataset[1000000:]
        assert len(submission_dataset) == 2000000

    __drop_dataset_duplicates(submission_dataset)

    return evaluation_dataset, submission_dataset


def calculate_metrics(evaluation_dataset, submission_dataset):
    submission_dataset["left_right"] = submission_dataset["left_instance_id"].astype(str) + submission_dataset[
        "right_instance_id"
    ].astype(str)
    predicted_values = submission_dataset["left_right"].values

    evaluation_dataset["left_right"] = evaluation_dataset["lid"].astype(str) + evaluation_dataset["rid"].astype(str)
    reference_values = evaluation_dataset["left_right"].values

    inter = set.intersection(set(predicted_values), set(reference_values))
    recall = len(inter) / len(reference_values)

    return round(recall, 3), len(inter), len(reference_values)


def save_not_matched_pairs(evaluation_dataset, submission_dataset, Z2_path):
    # Create composite keys for matching
    evaluation_dataset = evaluation_dataset.copy()
    evaluation_dataset["left_right"] = evaluation_dataset["lid"].astype(str) + evaluation_dataset["rid"].astype(str)
    reference_values = set(evaluation_dataset["left_right"].values)

    submission_dataset = submission_dataset.copy()
    submission_dataset["left_right"] = submission_dataset["left_instance_id"].astype(str) + submission_dataset[
        "right_instance_id"
    ].astype(str)
    predicted_values = set(submission_dataset["left_right"].values)

    # Find pairs that are in evaluation but NOT in submission
    not_matched_pairs = reference_values - predicted_values

    # Filter evaluation_dataset to get only the pairs (lid, rid) that weren't matched
    # This ensures we only save values from evaluation_dataset that are not in submission_dataset
    not_matched_eval = evaluation_dataset[evaluation_dataset["left_right"].isin(not_matched_pairs)].copy()

    # Load Z2 which contains id and name (sentence) columns
    Z2 = pd.read_csv(Z2_path)

    # Join with Z2 on left_id to get left_sentence
    result = not_matched_eval[["lid", "rid"]].copy()
    result = result.merge(Z2[["id", "name"]], left_on="lid", right_on="id", how="left")
    result = result.rename(columns={"name": "left_sentence"})
    result = result.drop(columns=["id"])

    # Join with Z2 on right_id to get right_sentence
    result = result.merge(Z2[["id", "name"]], left_on="rid", right_on="id", how="left")
    result = result.rename(columns={"name": "right_sentence", "lid": "left_id", "rid": "right_id"})
    result = result.drop(columns=["id"])

    # Reorder columns: left_id, left_sentence, right_id, right_sentence
    result = result[["left_id", "left_sentence", "right_id", "right_sentence"]]

    result.to_csv("not_matched_pairs.csv", index=False)


recalls = []
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate submission dataset")
    parser.add_argument(
        "--output-path",
        type=str,
        default="output.csv",
        help="Path to the submission dataset CSV file (default: output.csv)",
    )
    args = parser.parse_args()
    output_path = args.output_path

    f_measures = []
    base_path = "experiments/sigmod/hidden_data"
    # base_path = "experiments/sigmod/data"
    input_files = ["Y1.csv", "Y2.csv"]
    for i, eval_dataset in enumerate(input_files):
        evaluation_dataset_path = os.path.join(base_path, eval_dataset)

        evaluation_dataset, submission_dataset = get_evaluation_dataset_with_predicted_label(
            evaluation_dataset_path, output_path, dataset_id=i + 1
        )

        # Evaluate the submission
        recall, tp, all = calculate_metrics(evaluation_dataset, submission_dataset)
        Z2_path = os.path.join(base_path, "Z2.csv") if "hidden_data" in base_path else os.path.join(base_path, "X2.csv")
        save_not_matched_pairs(evaluation_dataset, submission_dataset, Z2_path)
        print(f"Recall for {eval_dataset} is {recall}.")
        recalls.append(recall)

    final_recall = round(np.mean(recalls), 3)
    print(f"Final recall is {final_recall}.")
