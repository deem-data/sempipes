import os
import time

import numpy as np
from EntityBlocking import block_x2
from FeatureExtracting import extract_x2
from FeatureExtracting_sempipes import extract_x2_sempipes
from x1_nn_regex import *

from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label
from experiments.sigmod.sustech.Main_optimizable import main_sempipes_optimizable_X2


def save_output(pairs_x1, expected_size_x1, pairs_x2, expected_size_x2, output_path):
    if len(pairs_x1) > expected_size_x1:
        pairs_x1 = pairs_x1[:expected_size_x1]
    elif len(pairs_x1) < expected_size_x1:
        pairs_x1.extend([(0, 0)] * (expected_size_x1 - len(pairs_x1)))
    if len(pairs_x2) > expected_size_x2:
        pairs_x2 = pairs_x2[:expected_size_x2]
    elif len(pairs_x2) < expected_size_x2:
        pairs_x2.extend([(0, 0)] * (expected_size_x2 - len(pairs_x2)))
    output = pairs_x1 + pairs_x2

    pd.DataFrame(pairs_x1, columns=["left_instance_id", "right_instance_id"]).to_csv(
        "output_x1_sustech.csv", index=False
    )

    output_df = pd.DataFrame(output, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv(output_path, index=False)


def evaluate(base_path, output_path):
    recalls = []
    input_files = ["Y1.csv", "Y2.csv"]
    for i, eval_dataset in enumerate(input_files):
        evaluation_dataset_path = os.path.join(base_path, eval_dataset)

        evaluation_dataset, submission_dataset = get_evaluation_dataset_with_predicted_label(
            evaluation_dataset_path, output_path, dataset_id=i + 1
        )

        # Evaluate the submission
        recall, tp, all = calculate_metrics(evaluation_dataset, submission_dataset)
        print(f"Recall for {eval_dataset} is {recall}.")
        recalls.append(recall)

    final_recall = round(np.mean(recalls), 3)
    return final_recall


def main_sempipes(data_path1, data_path2, path, save_path):
    # raw_data1 = pd.read_csv(data_path1)
    raw_data2 = pd.read_csv(data_path2)
    # raw_data1["title"] = raw_data1.title.str.lower()
    raw_data2["name"] = raw_data2.name.str.lower()

    # candidates_x1 = x1_test(raw_data1, 1000000, path)

    features = extract_x2_sempipes(raw_data2)
    candidates_x2 = block_x2(features, 2000000)

    candidates_x1 = pd.read_csv("output_x1_sustech.csv")
    candidates_x1 = candidates_x1.values.tolist()
    print("success")

    save_output(candidates_x1, 1000000, candidates_x2, 2000000, save_path)


def main_sempipes_optimizable_X2_full(data_path1, data_path2, path, save_path, mode):
    base_path_small = "experiments/sigmod/data"
    base_path_hidden = "experiments/sigmod/hidden_data"
    data_path_small2 = "experiments/sigmod/data/X2.csv"
    candidates_x2 = main_sempipes_optimizable_X2(data_path_small2, data_path2, mode, base_path_small, base_path_hidden)

    # raw_data1 = pd.read_csv(data_path1)
    # raw_data1["title"] = raw_data1.title.str.lower()

    # candidates_x1 = x1_test(raw_data1, 1000000, path)
    candidates_x1 = pd.read_csv("output_x1_sustech.csv")
    candidates_x1 = candidates_x1.values.tolist()
    print("success")

    save_output(candidates_x1, 1000000, candidates_x2, 2000000, save_path)


def main_manual(data_path1, data_path2, path, save_path):
    raw_data1 = pd.read_csv(data_path1)
    raw_data2 = pd.read_csv(data_path2)
    raw_data1["title"] = raw_data1.title.str.lower()
    raw_data2["name"] = raw_data2.name.str.lower()

    candidates_x1 = x1_test(raw_data1, 1000000, path)

    features = extract_x2(raw_data2)
    candidates_x2 = block_x2(features, 2000000)
    print("success")

    save_output(candidates_x1, 1000000, candidates_x2, 2000000, save_path)


def sempipes_full_pipeline(mode: int):
    if mode == 0:
        path = "experiments/sigmod/sempipes_sustech/fromstart_further_x1_berttiny_finetune_epoch20_margin0.01"
        data_path1 = "experiments/sigmod/hidden_data/Z1.csv"
        data_path2 = "experiments/sigmod/hidden_data/Z2.csv"
        save_path = "experiments/sigmod/results/sempipes/output_sempipes.csv"
        base_path = "experiments/sigmod/hidden_data"

    else:
        path = "experiments/sigmod/sempipes_sustech/fromstart_further_x1_berttiny_finetune_epoch20_margin0.01"
        data_path1 = "experiments/sigmod/data/X1.csv"
        data_path2 = "experiments/sigmod/data/X2.csv"
        save_path = "experiments/sigmod/results/sempipes/output_sempipes_small.csv"
        base_path = "experiments/sigmod/data"

    main_sempipes(data_path1, data_path2, path, save_path)
    recalls = evaluate(base_path, save_path)
    print(f"Evaluation is {np.mean(recalls)}.")

    return np.mean(recalls)


def optimized_full_pipeline_x2(mode: int):
    if mode == 0:
        path = "experiments/sigmod/sempipes_sustech/fromstart_further_x1_berttiny_finetune_epoch20_margin0.01"
        data_path1 = "experiments/sigmod/hidden_data/Z1.csv"
        data_path2 = "experiments/sigmod/hidden_data/Z2.csv"
        save_path = "experiments/sigmod/results/sempipes/output_sempipes_optimized.csv"
        base_path = "experiments/sigmod/hidden_data"

    else:
        path = "experiments/sigmod/sempipes_sustech/fromstart_further_x1_berttiny_finetune_epoch20_margin0.01"
        data_path1 = "experiments/sigmod/data/X1.csv"
        data_path2 = "experiments/sigmod/data/X2.csv"
        save_path = "experiments/sigmod/results/sempipes/output_sempipes_small_optimized.csv"
        base_path = "experiments/sigmod/data"

    main_sempipes_optimizable_X2_full(data_path1, data_path2, path, save_path, mode)
    recalls = evaluate(base_path, save_path)
    print(f"Evaluation is {np.mean(recalls)}.")

    return np.mean(recalls)


def manual_full_pipeline(mode: int):
    if mode == 0:
        path = "experiments/sigmod/sempipes_sustech/fromstart_further_x1_berttiny_finetune_epoch20_margin0.01"
        data_path1 = "experiments/sigmod/hidden_data/Z1.csv"
        data_path2 = "experiments/sigmod/hidden_data/Z2.csv"
        save_path = "experiments/sigmod/results/sempipes/output_manual.csv"
        base_path = "experiments/sigmod/hidden_data"

    else:
        path = "experiments/sigmod/sempipes_sustech/fromstart_further_x1_berttiny_finetune_epoch20_margin0.01"
        data_path1 = "experiments/sigmod/data/X1.csv"
        data_path2 = "experiments/sigmod/data/X2.csv"
        save_path = "experiments/sigmod/results/sempipes/output_manual_small.csv"
        base_path = "experiments/sigmod/data"

    main_manual(data_path1, data_path2, path, save_path)
    recalls = evaluate(base_path, save_path)
    print(f"Evaluation is {np.mean(recalls)}.")

    return np.mean(recalls)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    mode = 0  # 0 hidden, 1 small
    nreps = 5
    pipeline_name = "optimized"  # "manual", "sempipes", "optimized"
    t1 = time.time()

    recalls = []
    for i in range(nreps):
        if pipeline_name == "sempipes":
            recalls.append(sempipes_full_pipeline(mode))
        elif pipeline_name == "manual":
            recalls.append(manual_full_pipeline(mode))
        elif pipeline_name == "optimized":
            recalls.append(optimized_full_pipeline_x2(mode))

    print(f"Average recall: {np.mean(recalls)}")
    print(f"Standard deviation: {np.std(recalls)}")

    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
