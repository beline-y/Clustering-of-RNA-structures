"""Distance functions for RNA secondary structures.

Provides base-pair distance and Hamming distance, as well as utilities for
computing full pairwise distance matrices.
"""

from __future__ import annotations

from typing import List

import numpy as np
import RNA


def base_pair_distance(s1: str, s2: str) -> int:
    """Compute the base-pair distance between two dot-bracket structures.

    The base-pair distance counts the number of base pairs that are in one
    structure but not the other (symmetric set difference of base-pair sets).
    This is equivalent to ``RNA.bp_distance``.
    """
    return RNA.bp_distance(s1, s2)


def hamming_distance(s1: str, s2: str) -> int:
    """Compute the Hamming distance between two dot-bracket strings.

    Counts positions where the characters differ.
    """
    return sum(a != b for a, b in zip(s1, s2))


def pairwise_distance_matrix(
    structures: List[str], metric: str = "bp"
) -> np.ndarray:
    """Compute the full pairwise distance matrix for a list of structures.

    Parameters
    ----------
    structures:
        List of dot-bracket strings of equal length.
    metric:
        ``"bp"`` for base-pair distance, ``"hamming"`` for Hamming distance.

    Returns
    -------
    Symmetric (n x n) integer distance matrix.
    """
    n = len(structures)
    dist_fn = base_pair_distance if metric == "bp" else hamming_distance
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(structures[i], structures[j])
            D[i, j] = d
            D[j, i] = d
    return D
