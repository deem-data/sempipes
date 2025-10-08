# pylint: skip-file
import scanpy as sc
import scvi
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation


def preprocess_rna(
    adata,
    min_features: int = 600,
    min_cells: int = 3,
    target_sum: int = 10000,
    n_top_features=4000,  # or gene list
    is_hvg=True,
    batch_key="batchlb",
):
    
    adata.layers["counts"] = adata.X

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


def run_scVI(adata, batch_key, cell_type):
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_layers=8, n_latent=30, gene_likelihood="nb")

    model.train(max_epochs=100)

    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
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
        graph_connectivity=True, 
        kbet_per_label=True, 
        ilisi_knn=True, 
        pcr_comparison=True, 
        bras=True
    )

    names_obs = ["X_scVI"]
    print(names_obs)
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=cell_type_label,
        embedding_obsm_keys=names_obs,
        bio_conservation_metrics=_BIO_METRICS,
        batch_correction_metrics=_BATCH_METRICS,
        n_jobs=4,
    )
    bm.benchmark()
    a = bm.get_results(False, True)
    results = a.round(decimals=4)
    return results


adata_path = "comparisons/single_cell_analysis/data/ImmuneAtlas.h5ad"
epochs = 100
batch_key = "assay"
cell_type = "cell_type"

adata = sc.read(adata_path)

adata = run_scVI(adata, batch_key=batch_key, cell_type=cell_type)
results = evaluate_model(adata=adata, batch_key=batch_key, cell_type_label=cell_type)

print(results)

results.to_csv("comparisons/single_cell_analysis/results_unscaled_scvi.csv")
