"""Main script: Boltzmann sampling and clustering of RNA structures.

Runs the full pipeline for each RNA sequence defined in src/sequences.py:

 1. Sample 1000 structures from the Boltzmann ensemble (Vienna RNA pbacktrack).
 2. Compute base pair probabilities (BPPs) from samples and from the Vienna
    RNA partition function; compare them with a dot plot.
 3. Compute pairwise base-pair distances and embed in 2D using MDS.
 4. Cluster with k-means; pick k via silhouette score.
 5. Compute per-cluster BPPs, centroid structures and MEA structures.
 6. Analyse convergence (RMSD vs Vienna BPPs as a function of sample size).

Results are saved to the ``results/<rna_name>/`` directory.
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from src.sequences import SEQUENCES
from src.sampling import (
    sample_structures,
    get_vienna_bpp,
    compute_bpp_from_samples,
    get_mfe_structure,
)
from src.distances import pairwise_distance_matrix
from src.visualization import (
    plot_comparison_dot_plot,
    plot_combined_dot_plot,
    plot_2d_embedding,
    plot_cluster_dot_plots,
    plot_convergence,
)
from src.clustering import (
    embed_mds,
    embed_tsne,
    cluster_kmeans,
    determine_optimal_k,
    plot_elbow,
)
from src.cluster_analysis import (
    compute_cluster_probabilities,
    compute_cluster_bpp,
    compute_centroid_structure,
    compute_all_centroid_structures,
)
from src.mea import compute_mea_structure
from src.convergence import analyze_convergence

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
N_SAMPLES = 1000          # samples per RNA
RESULTS_DIR = "results"
SEED = 42


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
#  Per-RNA pipeline
# --------------------------------------------------------------------------- #
def run_rna(name: str, info: dict) -> None:
    sequence = info["sequence"]
    description = info["description"]
    n_clusters = info.get("n_clusters", 3)
    out_dir = ensure_dir(os.path.join(RESULTS_DIR, name))

    print_section(f"{name}  –  {description}")
    print(f"  Sequence  : {sequence}")
    print(f"  Length    : {len(sequence)} nt")

    # ------------------------------------------------------------------ #
    #  1. Sample structures
    # ------------------------------------------------------------------ #
    print(f"\n[1] Sampling {N_SAMPLES} structures …")
    structures = sample_structures(sequence, N_SAMPLES, seed=SEED)
    print(f"    Sampled {len(structures)} structures.")

    # ------------------------------------------------------------------ #
    #  2. Base pair probabilities
    # ------------------------------------------------------------------ #
    print("[2] Computing base pair probabilities …")
    vienna_bpp = get_vienna_bpp(sequence)
    sampled_bpp = compute_bpp_from_samples(structures)
    mfe_struct, mfe_energy = get_mfe_structure(sequence)

    from src.convergence import rmsd_bpp
    rmsd_val = rmsd_bpp(sampled_bpp, vienna_bpp)
    print(f"    MFE structure : {mfe_struct}  ({mfe_energy:.2f} kcal/mol)")
    print(f"    BPP RMSD (sampled vs Vienna) : {rmsd_val:.4f}")

    # Dot plots
    plot_comparison_dot_plot(
        sampled_bpp, vienna_bpp, name,
        output_path=os.path.join(out_dir, "dot_plot_comparison.png"),
    )
    plot_combined_dot_plot(
        sampled_bpp, vienna_bpp, name,
        output_path=os.path.join(out_dir, "dot_plot_combined.png"),
    )
    plt.close("all")
    print("    Dot plots saved.")

    # ------------------------------------------------------------------ #
    #  3. Pairwise distances & MDS embedding
    # ------------------------------------------------------------------ #
    print("[3] Computing pairwise distances …")
    dist_matrix = pairwise_distance_matrix(structures, metric="bp")
    print(f"    Mean BP distance : {dist_matrix.mean():.1f}")

    print("    Embedding with MDS …")
    mds_coords = embed_mds(dist_matrix)

    # ------------------------------------------------------------------ #
    #  4. Clustering
    # ------------------------------------------------------------------ #
    print(f"[4] Clustering (k-means, k={n_clusters}) …")
    best_k, ks_arr, inertias, silhouettes = determine_optimal_k(mds_coords, max_k=8)
    plot_elbow(
        ks_arr,
        inertias,
        silhouettes,
        name,
        output_path=os.path.join(out_dir, "elbow_plot.png"),
    )
    print(f"    Best k by silhouette : {best_k}")
    # Use user-specified k if provided, otherwise best_k
    k = n_clusters
    labels, centers = cluster_kmeans(mds_coords, k)

    # Visualise MDS embedding
    plot_2d_embedding(
        mds_coords, labels, name, method="MDS",
        output_path=os.path.join(out_dir, "mds_clusters.png"),
    )
    plt.close("all")
    print(f"    Cluster sizes: { {lbl: int((labels==lbl).sum()) for lbl in range(k)} }")

    # t-SNE embedding (only for larger datasets)
    if len(structures) >= 50:
        try:
            print("    Embedding with t-SNE …")
            tsne_coords = embed_tsne(dist_matrix)
            plot_2d_embedding(
                tsne_coords, labels, name, method="t-SNE",
                output_path=os.path.join(out_dir, "tsne_clusters.png"),
            )
            plt.close("all")
        except Exception as exc:
            print(f"    t-SNE skipped: {exc}")

    # ------------------------------------------------------------------ #
    #  5. Per-cluster analysis
    # ------------------------------------------------------------------ #
    print("[5] Per-cluster BPPs, centroid and MEA structures …")
    cluster_probs = compute_cluster_probabilities(labels)
    cluster_bpps = compute_cluster_bpp(structures, labels)
    centroid_structs = compute_all_centroid_structures(cluster_bpps)

    print(f"    Cluster probabilities:")
    for lbl, prob in sorted(cluster_probs.items()):
        centroid = centroid_structs[lbl]
        mea_struct, ea = compute_mea_structure(cluster_bpps[lbl])
        print(f"      Cluster {lbl} : {prob:.2%}")
        print(f"        Centroid : {centroid}")
        print(f"        MEA      : {mea_struct}  (EA={ea:.3f})")

    # Per-cluster dot plots
    plot_cluster_dot_plots(
        cluster_bpps, cluster_probs, name,
        output_path=os.path.join(out_dir, "cluster_dot_plots.png"),
    )
    plt.close("all")
    print("    Cluster dot plots saved.")

    # ------------------------------------------------------------------ #
    #  6. Convergence analysis
    # ------------------------------------------------------------------ #
    print("[6] Convergence analysis …")
    ns, rmsds = analyze_convergence(structures, vienna_bpp)
    plot_convergence(
        ns, rmsds, name,
        output_path=os.path.join(out_dir, "convergence.png"),
    )
    plt.close("all")
    print(f"    Convergence plot saved.  Final RMSD at {N_SAMPLES} samples: {rmsds[-1]:.4f}")

    # ------------------------------------------------------------------ #
    #  Save summary
    # ------------------------------------------------------------------ #
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(f"RNA: {name}\n")
        fh.write(f"Description: {description}\n")
        fh.write(f"Sequence: {sequence}\n")
        fh.write(f"Length: {len(sequence)} nt\n")
        fh.write(f"Samples: {N_SAMPLES}\n")
        fh.write(f"MFE structure: {mfe_struct}  ({mfe_energy:.2f} kcal/mol)\n")
        fh.write(f"BPP RMSD (sampled vs Vienna): {rmsd_val:.4f}\n")
        fh.write(f"Clusters (k={k}):\n")
        for lbl, prob in sorted(cluster_probs.items()):
            centroid = centroid_structs[lbl]
            mea_struct, ea = compute_mea_structure(cluster_bpps[lbl])
            fh.write(f"  Cluster {lbl} ({prob:.2%}): centroid={centroid}  mea={mea_struct}\n")
    print(f"    Summary written to {summary_path}")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ensure_dir(RESULTS_DIR)
    names = sys.argv[1:] if len(sys.argv) > 1 else list(SEQUENCES.keys())
    for rna_name in names:
        if rna_name not in SEQUENCES:
            print(f"Unknown RNA '{rna_name}'. Available: {list(SEQUENCES.keys())}")
            continue
        run_rna(rna_name, SEQUENCES[rna_name])
    print("\nDone. Results saved to", RESULTS_DIR)
