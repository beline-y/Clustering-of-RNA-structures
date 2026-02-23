"""Cluster analysis: probabilities, per-cluster BPPs, and centroid structures.

Given a list of sampled structures and their cluster labels, this module
computes:
  - cluster probabilities (fraction of structures in each cluster)
  - per-cluster base pair probabilities (BPPs)
  - centroid representative structure (all base pairs with conditional
    probability > 0.5 in the cluster)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .sampling import dot_bracket_to_pairs, pairs_to_dot_bracket


def compute_cluster_probabilities(labels: np.ndarray) -> Dict[int, float]:
    """Return the fraction of structures assigned to each cluster.

    Parameters
    ----------
    labels:
        Array of integer cluster labels of length n_structures.

    Returns
    -------
    dict mapping cluster label → probability.
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {int(lbl): count / total for lbl, count in zip(unique, counts)}


def compute_cluster_bpp(
    structures: List[str], labels: np.ndarray
) -> Dict[int, np.ndarray]:
    """Compute per-cluster base pair probability matrices.

    For each cluster *c*, computes the conditional BPP: the fraction of
    structures in cluster *c* that contain each base pair.

    Parameters
    ----------
    structures:
        List of dot-bracket strings.
    labels:
        Cluster label for each structure.

    Returns
    -------
    dict mapping cluster label → (n x n) BPP matrix.
    """
    n = len(structures[0])
    cluster_bpps: Dict[int, np.ndarray] = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        cluster_structs = [s for s, m in zip(structures, mask) if m]
        counts = np.zeros((n, n))
        for struct in cluster_structs:
            for i, j in dot_bracket_to_pairs(struct):
                counts[i, j] += 1
                counts[j, i] += 1
        cluster_bpps[int(lbl)] = counts / len(cluster_structs)
    return cluster_bpps


def compute_centroid_structure(
    cluster_bpp: np.ndarray, threshold: float = 0.5
) -> str:
    """Compute the centroid representative structure for a cluster.

    Selects all base pairs (i, j) with conditional probability above
    *threshold* and returns them as a dot-bracket string.

    This always produces a valid nested structure because the conditional
    probabilities of all partners for a single position sum to at most 1:
    if P(i,j | c) > 0.5 and P(i,k | c) > 0.5 for k ≠ j, their sum would
    exceed 1 — a contradiction.  Hence at most one partner per position can
    exceed the 0.5 threshold.

    Parameters
    ----------
    cluster_bpp:
        (n x n) conditional BPP matrix for the cluster.
    threshold:
        Probability threshold (default 0.5).

    Returns
    -------
    Dot-bracket structure string.
    """
    n = cluster_bpp.shape[0]
    pairs: List[Tuple[int, int]] = []
    paired = set()
    # Only examine upper triangle to avoid counting pairs twice
    rows, cols = np.where(np.triu(cluster_bpp, k=1) > threshold)
    for i, j in zip(rows, cols):
        if i not in paired and j not in paired:
            pairs.append((int(i), int(j)))
            paired.add(int(i))
            paired.add(int(j))
    return pairs_to_dot_bracket(pairs, n)


def compute_all_centroid_structures(
    cluster_bpps: Dict[int, np.ndarray], threshold: float = 0.5
) -> Dict[int, str]:
    """Return centroid structures for all clusters."""
    return {
        lbl: compute_centroid_structure(bpp, threshold)
        for lbl, bpp in cluster_bpps.items()
    }
