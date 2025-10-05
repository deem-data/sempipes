# pylint: skip-file
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def preprocess_rna(
    adata,
    min_features: int = 600,
    min_cells: int = 3,
    target_sum: int = 10000,
    n_top_features=2000,  # or gene list
    is_hvg=True,
    batch_key="batchlb",
):
    if min_features is None:
        min_features = 600
    if n_top_features is None:
        n_top_features = 40000

    adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(["ERCC", "MT-", "mt-"]))]]

    cells_subset, _ = sc.pp.filter_cells(adata, min_genes=min_features, inplace=False)
    adata = adata[cells_subset, :]
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if is_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=True, subset=True)

    print("Processed dataset shape: {}".format(adata.shape))
    return adata


def train_qr_scVI(adata_ref, adata_query, batch_key, cell_type):
    scvi.model.SCVI.setup_anndata(adata_ref, batch_key=batch_key)
    model = scvi.model.SCVI(adata_ref, n_layers=8, n_latent=30, gene_likelihood="nb")

    model.train(max_epochs=100)

    embedding = model.get_latent_representation()

    # Query
    scvi.model.SCVI.prepare_query_anndata(adata_query, model, return_reference_var_names=True)
    scvi_query = scvi.model.SCVI.load_query_data(adata_query, model)

    scvi_query.train(max_epochs=100, accelerator="cpu")
    embedding_test = scvi_query.get_latent_representation()

    # Evaluate
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embedding, adata_ref.obs[cell_type].tolist())
    cat_preds = knn.predict(embedding_test)

    cell_types_list = pd.unique(adata_query.obs[cell_type]).tolist()
    acc = accuracy_score(adata_query.obs[cell_type].to_list(), cat_preds)
    f1 = f1_score(adata_query.obs[cell_type].to_list(), cat_preds, labels=cell_types_list, average=None)
    f1_weighted = f1_score(adata_query.obs[cell_type].to_list(), cat_preds, labels=cell_types_list, average="weighted")
    f1_macro = f1_score(adata_query.obs[cell_type].to_list(), cat_preds, labels=cell_types_list, average="macro")
    f1_median = np.median(f1)

    print(f"Per class {cell_types_list} F1 {f1}")
    print(
        "Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} ".format(
            acc, f1_median, f1_macro, f1_weighted
        ),
    )

    return acc, f1_macro


adata_path = "comparisons/cell_typing/data/ImmuneAtlas_raw.h5ad"
batch_key = "assay"
cell_type = "cell_type"

adata = sc.read(adata_path)
adata = preprocess_rna(adata=adata, is_hvg=True, batch_key=batch_key)


adata_ref = adata[adata.obs[batch_key] != "10x 5' v2",].copy()
adata_query = adata[adata.obs[batch_key] == "10x 5' v2",].copy()


accs, macro_f1s = [], []

for i in range(0, 5):
    acc, macro_f1 = train_qr_scVI(
        adata_ref=adata_ref, adata_query=adata_query, batch_key=batch_key, cell_type=cell_type
    )
    accs.append(acc)
    macro_f1s.append(macro_f1)

print(f"After 5 runs AVG: Accuracy {np.mean(accs).round(3)}, macro F1 {np.mean(macro_f1s).round(3)}")
print(f"After 5 runs ATD: Accuracy {np.std(accs).round(3)}, macro F1 {np.std(macro_f1s).round(3)}")
