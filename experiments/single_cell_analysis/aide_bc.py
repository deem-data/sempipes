# pylint: skip-file
import harmonypy as harmony
import scanpy as sc
import scib

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

# Batch correction with Harmony
harmony_integrated = harmony.run_harmony(adata.obsm["X_pca"], adata.obs, "batch")
adata.obsm["X_pca_harmony"] = harmony_integrated.Z_corr.T

# Calculate metrics
results = scib.metrics.metrics(
    adata,
    batch_key="batch",
    label_key="cell_type",
    embed="X_pca_harmony",
    isolated_labels=None,
    cluster=False,
)

print("Batch score:", results["batch"])
print("Bio conservation score:", results["bio"])
print("Overall score:", results["total"])
