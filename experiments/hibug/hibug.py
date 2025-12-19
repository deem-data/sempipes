# https://github.com/cure-lab/HiBug/blob/main/celebA_use_case.ipynb
import csv
import json

import numpy as np

f = open("experiments/hibug/celeba/list_attr_celeba.txt")
lines = list(f.readlines())
lines = [line.strip("\n").split(" ")[0] for line in lines]
all_datas = ["experiments/hibug/celeba/img_align_celeba/" + line for line in lines][1:]
train_idx = np.load("experiments/hibug/celeba/train_idx.npy")[:80000]
valid_idx = np.arange(len(all_datas))[-100000:-80000]
unlabel_idx = np.arange(len(all_datas))[-80000:]
idxs = np.concatenate([train_idx, valid_idx, unlabel_idx], axis=0)
labels = np.load("experiments/hibug/celeba/labels.npy")[idxs]
predictions = np.load("experiments/hibug/celeba/predictions.npy")[idxs]
all_datas = [all_datas[i] for i in idxs]
attributes = json.load(open("experiments/hibug/celeba/attribute.json"))

attribute_names = list(attributes.keys())


with open("experiments/hibug/hibug_attributes.csv", mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["idx", "image", "label", "prediction"] + attribute_names
    writer.writerow(header)
    for idx in valid_idx:
        row = [
            idx,
            f"experiments/hibug/celeba/img_align_celeba/{lines[idx]}",
            labels[idx],
            predictions[idx],
        ]
        for attribute in attribute_names:
            if "data" in attributes[attribute]:
                row.append(attributes[attribute]["data"][idx])
        writer.writerow(row)
