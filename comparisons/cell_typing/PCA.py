# pylint: skip-file
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def train_qr(adata_ref, adata_query, batch_key, cell_type):
    # Apply PCA
    pca = PCA()
    X_pca_ref = pca.fit_transform(adata_ref.X)
    X_pca_query = pca.transform(adata_query.X)

    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_pca_ref, adata_ref.obs[cell_type].tolist())

    # Make predictions
    cat_preds = knn.predict(X_pca_query)

    cell_types_list = pd.unique(adata_query.obs[cell_type]).tolist()
    acc = accuracy_score(adata_query.obs[cell_type].to_list(), cat_preds)
    f1_macro = f1_score(adata_query.obs[cell_type].to_list(), cat_preds, labels=cell_types_list, average="macro")

    return acc, f1_macro


adata_path_ref = "comparisons/cell_typing/data/ImmuneAtlas_raw_train.h5ad"
adata_path_query = "comparisons/cell_typing/data/ImmuneAtlas_raw_test.h5ad"
adata_ref = sc.read(adata_path_ref)
adata_query = sc.read(adata_path_query)
batch_key = "batchlb"
cell_type = "cell_type"

accs, macro_f1s = [], []

# for i in range(0, 5):
acc, macro_f1 = train_qr(adata_ref=adata_ref, adata_query=adata_query, batch_key=batch_key, cell_type=cell_type)
accs.append(acc)
macro_f1s.append(macro_f1)

print(f"After 5 runs AVG: Accuracy {np.mean(accs).round(3)}, macro F1 {np.mean(macro_f1s).round(3)}")
print(f"After 5 runs ATD: Accuracy {np.std(accs).round(3)}, macro F1 {np.std(macro_f1s).round(3)}")
