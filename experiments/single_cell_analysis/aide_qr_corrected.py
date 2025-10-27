# pylint: skip-file
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load the data
adata_full = anndata.read_h5ad("comparisons/single_cell_analysis/data/ImmuneAtlas_raw.h5ad")

# Preprocessing data
sc.pp.normalize_total(adata_full, target_sum=1e4)
sc.pp.log1p(adata_full)
sc.pp.highly_variable_genes(adata_full, n_top_genes=2000)
adata_full = adata_full[:, adata_full.var.highly_variable]

adata = adata_full[adata_full.obs["assay"] != "10x 5' v2",].copy()
adata_test = adata_full[adata_full.obs["assay"] == "10x 5' v2",].copy()


accuracies, f1_scores = [], []
for run in range(5):
    # Prepare features and target
    X_train = adata.X
    y_train = adata.obs["cell_type"]
    X_test = adata_test.X
    y_test = adata_test.obs["cell_type"]

    # Initialize and train the Random Forest model with increased number of trees
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    f1 = f1_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Macro F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

    accuracies.append(accuracy)
    f1_scores.append(f1)

print(f"After 5 runs AVG: Accuracy {np.mean(accuracies).round(3)}, macro F1 {np.mean(f1_scores).round(3)}")
print(f"After 5 runs ATD: Accuracy {np.std(accuracies).round(3)}, macro F1 {np.std(f1_scores).round(3)}")
