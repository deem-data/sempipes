import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from harmony import harmonize
from sklearn.metrics import silhouette_score
from anndata import AnnData
import scanpy as sc

from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

def evaluate_model(adata, batch_key, cell_type_label):
    _BIO_METRICS = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_leiden=True,
        nmi_ari_cluster_labels_kmeans=False,
        silhouette_label=True,
        clisi_knn=True,
    )
    _BATCH_METRICS = BatchCorrection(
        graph_connectivity=True, 
        kbet_per_label=True, 
        ilisi_knn=True, 
        pcr_comparison=True, 
        bras=True
    )
    
    names_obsm = ["X_pca_harmony"]
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


# Load the data
adata_path = "comparisons/single_cell_analysis/data/ImmuneAtlas_raw.h5ad"
adata = sc.read(adata_path)

# Normalize and log-transform the data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Run PCA
sc.tl.pca(adata, svd_solver="arpack")

# Harmonize the data across batches
batch_key = "assay"
cell_type = "cell_type"

adata.obsm["X_pca_harmony"] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key)

# Compute scores for batch correction

results = evaluate_model(adata=adata, batch_key=batch_key, cell_type_label=cell_type)
print(results)

results.to_csv("comparisons/single_cell_analysis/results/esults_unscaled_aide.csv")