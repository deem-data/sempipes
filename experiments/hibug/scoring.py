import numpy as np
import pandas as pd


def score_sempipes(attributes):
    sempipes_matches = [
        ("Male", "gender", "male"),
        ("Heavy_Makeup", "makeup", "yes"),
        ("Pale_Skin", "skin_color", "white"),
        ("Black_Hair", "hair_color", "black"),
        ("Brown_Hair", "hair_color", "brown"),
        ("Blond_Hair", "hair_color", "blonde"),
        ("No_Beard", "beard", "no"),
        ("Young", "age", "young"),
    ]

    accuracy, std = score(attributes, sempipes_matches)
    return accuracy, std


def score_sempipes_optimized(attributes):
    sempipes_matches = [
        ("Male", "gender", "male"),
        ("Heavy_Makeup", "makeup", "yes"),
        ("Pale_Skin", "skin_color", "light"),
        ("Black_Hair", "hair_color", "black"),
        ("Brown_Hair", "hair_color", "brown"),
        ("Blond_Hair", "hair_color", "blonde"),
        ("No_Beard", "beard", "no"),
        ("Young", "age", "young"),
    ]

    accuracy, std = score(attributes, sempipes_matches)
    return accuracy, std


def score_hibug():
    hibug_matches = [
        ("Male", "gender", "Male"),
        ("Heavy_Makeup", "makeup", "Yes"),
        ("Pale_Skin", "skin", "white"),
        ("Black_Hair", "hair color", "black"),
        ("Brown_Hair", "hair color", "brown"),
        ("Blond_Hair", "hair color", "blonde"),
        ("No_Beard", "beard", "No"),
        ("Young", "young", "young"),
    ]

    hibug = pd.read_csv("experiments/hibug/hibug_attributes.csv")
    hibug.image = hibug.image.str.replace("experiments/hibug/img_align_celeba/", "")
    # First 2000 rows are heldout for optimizer
    hibug = hibug.iloc[2000:]
    accuracy, std = score(hibug, hibug_matches)
    return accuracy, std


def score(attributes, matches):
    groundtruth = pd.read_csv("experiments/hibug/celeba/list_attr_celeba.csv")
    attributes.image = attributes.image.str.replace("experiments/hibug/img_align_celeba/", "")
    to_compare = groundtruth.merge(attributes, left_on="image_id", right_on="image")
    return annotation_accuracy(to_compare, matches)


def annotation_accuracy(to_compare, matches):
    accuracies = []
    for gt_column, annotated_column, annotated_value in matches:
        accuracy = np.sum(
            ((to_compare[gt_column] == 1) & (to_compare[annotated_column] == annotated_value))
            | ((to_compare[gt_column] == -1) & (to_compare[annotated_column] != annotated_value))
        ) / len(to_compare)
        # print(f"\tAccuracy for {gt_column}: {accuracy:.4f}")
        accuracies.append(accuracy)
    return np.mean(accuracies), np.std(accuracies)


if __name__ == "__main__":
    sempipes = pd.read_csv("experiments/hibug/sempipes_attributes.csv")
    sempipes_optimized = pd.read_csv("experiments/hibug/sempipes_optimized_attributes.csv")

    print("Hibug annotation accuracy:", score_hibug())
    print("Sempipes annotation accuracy:", score_sempipes(sempipes))
    print("Sempipes optimized annotation accuracy:", score_sempipes_optimized(sempipes_optimized))
