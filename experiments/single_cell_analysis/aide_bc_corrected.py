# pylint: skip-file
import harmonypy as harmony
import scanpy as sc
import scib
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
        graph_connectivity=True, kbet_per_label=True, ilisi_knn=True, pcr_comparison=True, bras=True
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
adata = sc.read_h5ad("comparisons/single_cell_analysis/data/ImmuneHuman.h5ad")

# Preprocess the data
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

# Run PCA
sc.tl.pca(adata, svd_solver="arpack")

batch_key = "assay"
cell_type = "cell_type"

# Batch correction with Harmony
harmony_integrated = harmony.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key)
adata.obsm["X_pca_harmony"] = harmony_integrated.Z_corr.T

# Compute scores for batch correction
results = evaluate_model(adata=adata, batch_key=batch_key, cell_type_label=cell_type)
print(results)

results.to_csv("comparisons/single_cell_analysis/results/results_unscaled_aide.csv")
