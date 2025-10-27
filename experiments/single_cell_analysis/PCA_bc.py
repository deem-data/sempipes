# pylint: skip-file
import numpy as np
import scanpy as sc
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from sklearn.decomposition import PCA


def get_representation(adata, batch_key, cell_type):
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(adata.X)

    adata.obsm["pca"] = X_pca

    return adata


def evaluate_model(adata, batch_key, cell_type_label):
    _BIO_METRICS = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_leiden=True,
        nmi_ari_cluster_labels_kmeans=False,
        silhouette_label=True,
        clisi_knn=True,
    )
    _BATCH_METRICS = BatchCorrection(
        graph_connectivity=True, kbet_per_label=True, ilisi_knn=True, pcr_comparison=True, bras=True
    )

    names_obsm = ["pca"]
    print(names_obsm)
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=cell_type_label,
        embedding_obsm_keys=names_obsm,
        bio_conservation_metrics=_BIO_METRICS,
        batch_correction_metrics=_BATCH_METRICS,
        n_jobs=4,
    )
    bm.benchmark()
    a = bm.get_results(False, True)
    results = a.round(decimals=4)
    return results


adata_path = "comparisons/single_cell_analysis/data/ImmuneAtlas_raw.h5ad"
adata = sc.read(adata_path)
batch_key = "batchlb"
cell_type = "cell_type"

adata = get_representation(adata, batch_key=batch_key, cell_type=cell_type)
results = evaluate_model(adata=adata, batch_key=batch_key, cell_type_label=cell_type)

print(results)

results.to_csv("comparisons/single_cell_analysis/results_unscaled_pca.csv")
