"""Clustering of RNA secondary structures.

Provides dimensionality reduction (MDS, t-SNE) and k-means clustering
applied to pairwise distance matrices computed from sampled structures.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score


def embed_mds(dist_matrix: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    """Embed structures in low-dimensional space using Multidimensional Scaling.

    Parameters
    ----------
    dist_matrix:
        Square symmetric pairwise distance matrix.
    n_components:
        Target dimensionality (default 2 for plotting).
    seed:
        Random seed.

    Returns
    -------
    (n_structures x n_components) coordinate array.
    """
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
    )
    return mds.fit_transform(dist_matrix)


def embed_tsne(dist_matrix: np.ndarray, seed: int = 42) -> np.ndarray:
    """Embed structures using t-SNE on the precomputed distance matrix.

    Returns a (n_structures x 2) coordinate array.
    """
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        random_state=seed,
        init="random",
        perplexity=min(30, len(dist_matrix) // 4),
    )
    return tsne.fit_transform(dist_matrix.astype(float))


def cluster_kmeans(
    coords: np.ndarray, k: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Run k-means clustering on a 2-D coordinate array.

    Parameters
    ----------
    coords:
        (n x d) coordinate array (e.g. from MDS/t-SNE).
    k:
        Number of clusters.
    seed:
        Random seed.

    Returns
    -------
    labels : (n,) cluster-label array.
    centers : (k x d) cluster-centre array.
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(coords)
    return labels, km.cluster_centers_


def determine_optimal_k(
    coords: np.ndarray, max_k: int = 8, seed: int = 42
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Determine the optimal number of clusters using the elbow method
    (inertia) and silhouette score.

    Parameters
    ----------
    coords:
        (n x d) coordinate array.
    max_k:
        Maximum number of clusters to test.
    seed:
        Random seed.

    Returns
    -------
    best_k : Chosen k based on highest silhouette score.
    ks : k values tested (2 .. max_k).
    inertias : Inertia values for k = 2 .. max_k.
    silhouettes : Silhouette scores for k = 2 .. max_k.
    """
    max_k = min(max_k, len(coords) - 1)
    ks = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(coords)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(coords, labels))
    best_k = ks[int(np.argmax(silhouettes))]
    return best_k, np.array(ks), np.array(inertias), np.array(silhouettes)


def plot_elbow(
    ks: np.ndarray,
    inertias: np.ndarray,
    silhouettes: np.ndarray,
    rna_name: str,
    output_path: str | None = None,
) -> None:
    """Plot inertia and silhouette score vs k to aid cluster-number selection."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, inertias, marker="o")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow plot")

    ax2.plot(ks, silhouettes, marker="o", color="orange")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette scores")

    fig.suptitle(f"Cluster selection – {rna_name}")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
