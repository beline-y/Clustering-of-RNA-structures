"""Convergence analysis for Boltzmann-sampled base pair probabilities.

Studies how well the sampled BPP matrix converges to the true (partition
function) BPPs as the number of samples grows.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .sampling import compute_bpp_from_samples


def rmsd_bpp(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square deviation between two BPP matrices (upper triangle only)."""
    n = a.shape[0]
    idx = np.triu_indices(n, k=1)
    return float(np.sqrt(np.mean((a[idx] - b[idx]) ** 2)))


def analyze_convergence(
    structures: List[str],
    vienna_bpp: np.ndarray,
    checkpoints: List[int] | None = None,
) -> Tuple[List[int], List[float]]:
    """Compute BPP RMSD vs Vienna RNA BPPs at increasing sample sizes.

    Parameters
    ----------
    structures:
        List of sampled dot-bracket strings (in sampling order).
    vienna_bpp:
        Reference BPP matrix from the partition function.
    checkpoints:
        List of sample counts at which to evaluate RMSD.  Defaults to
        logarithmically spaced values from 10 to ``len(structures)``.

    Returns
    -------
    ns : Sample counts evaluated.
    rmsds : Corresponding RMSD values.
    """
    n_total = len(structures)
    if checkpoints is None:
        checkpoints = sorted(
            set(
                int(x)
                for x in np.logspace(1, np.log10(n_total), num=20).round()
                if 1 < x <= n_total
            )
        )
        if n_total not in checkpoints:
            checkpoints.append(n_total)

    ns: List[int] = []
    rmsds: List[float] = []
    for n in checkpoints:
        sampled_bpp = compute_bpp_from_samples(structures[:n])
        ns.append(n)
        rmsds.append(rmsd_bpp(sampled_bpp, vienna_bpp))
    return ns, rmsds
