"""Maximum Expected Accuracy (MEA) structure computation.

Implements a Nussinov-type dynamic programming algorithm to find the structure
*R* that maximises the expected accuracy

    EA(R) = Σ_{(i,j)∈R} 2·γ·p_{ij}  +  Σ_{i unpaired in R} pu_i

where ``p_{ij}`` are the base pair probabilities and
``pu_i = 1 - Σ_j p_{ij}`` is the probability that nucleotide *i* is unpaired.

The parameter *γ* (default 1) weights paired vs unpaired contributions.
Setting γ > 0.5 favours more base pairs; γ < 0.5 penalises them.

Reference: Do, C.B., Woods, D.A. & Batzoglou, S. (2006) CONTRAfold.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .sampling import pairs_to_dot_bracket


def compute_mea_structure(
    bpp_matrix: np.ndarray, gamma: float = 1.0
) -> Tuple[str, float]:
    """Compute the MEA structure using Nussinov-type dynamic programming.

    Parameters
    ----------
    bpp_matrix:
        Symmetric (n x n) base pair probability matrix (0-indexed).
    gamma:
        Pairing weight parameter (default 1.0).

    Returns
    -------
    mea_structure : Dot-bracket string of the MEA structure.
    ea_score : Expected accuracy of the returned structure.
    """
    n = bpp_matrix.shape[0]

    # Unpaired probability for each position
    pu = 1.0 - bpp_matrix.sum(axis=1)
    pu = np.clip(pu, 0.0, 1.0)

    # DP table: opt[i][j] = max EA for subsequence i..j (0-indexed, inclusive)
    opt = np.zeros((n, n))
    # Traceback table: tb[i][j] = -1 (i unpaired) or k (i pairs with k)
    tb = np.full((n, n), -1, dtype=int)

    for length in range(1, n):
        for i in range(n - length):
            j = i + length

            # Option 1: position i is unpaired → add pu[i]
            best = opt[i + 1, j] + pu[i]
            best_tb = -1

            # Option 2: i pairs with some k in [i+1, j]
            for k in range(i + 1, j + 1):
                p_ik = bpp_matrix[i, k]
                if p_ik == 0:
                    continue
                inner = opt[i + 1, k - 1] if k > i + 1 else 0.0
                outer = opt[k + 1, j] if k < j else 0.0
                score = inner + 2.0 * gamma * p_ik + outer
                if score > best:
                    best = score
                    best_tb = k

            opt[i, j] = best
            tb[i, j] = best_tb

    # Traceback
    pairs = _traceback(tb, 0, n - 1)
    structure = pairs_to_dot_bracket(pairs, n)
    return structure, float(opt[0, n - 1])


def _traceback(
    tb: np.ndarray, i: int, j: int
) -> List[Tuple[int, int]]:
    """Recursively traceback the MEA DP table to recover base pairs."""
    if i >= j:
        return []
    k = tb[i, j]
    if k == -1:
        # i unpaired
        return _traceback(tb, i + 1, j)
    else:
        inner = _traceback(tb, i + 1, k - 1) if k > i + 1 else []
        outer = _traceback(tb, k + 1, j) if k < j else []
        return inner + [(i, k)] + outer
