"""Boltzmann sampling of RNA structures and base pair probability computation.

Uses the Vienna RNA package (ViennaRNA Python bindings).  The fold compound
must be created with ``md.uniq_ML = 1`` to enable stochastic traceback via
``fc.pbacktrack()``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import RNA


def _make_fold_compound(sequence: str) -> RNA.fold_compound:
    """Create a fold compound with unique multiloop decomposition enabled."""
    md = RNA.md()
    md.uniq_ML = 1
    return RNA.fold_compound(sequence, md)


def sample_structures(sequence: str, n_samples: int, seed: int = 42) -> List[str]:
    """Sample RNA secondary structures from the Boltzmann ensemble.

    Parameters
    ----------
    sequence:
        RNA sequence string (uppercase, U not T).
    n_samples:
        Number of structures to sample.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of dot-bracket structure strings.
    """
    fc = _make_fold_compound(sequence)
    fc.pf()
    structures = fc.pbacktrack(n_samples)
    return list(structures)


def get_vienna_bpp(sequence: str) -> np.ndarray:
    """Compute base pair probabilities using the Vienna RNA partition function.

    Returns a symmetric (n x n) matrix where ``bpp[i, j]`` is the probability
    that nucleotides *i* and *j* (0-indexed) form a base pair.
    """
    n = len(sequence)
    fc = _make_fold_compound(sequence)
    fc.pf()
    raw_bpp = fc.bpp()  # 1-indexed tuple-of-tuples; raw_bpp[i][j] for 1<=i<j<=n

    bpp_matrix = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = raw_bpp[i][j]
            if p > 0:
                bpp_matrix[i - 1, j - 1] = p
                bpp_matrix[j - 1, i - 1] = p
    return bpp_matrix


def get_mfe_structure(sequence: str) -> Tuple[str, float]:
    """Return the MFE structure and its free energy."""
    fc = _make_fold_compound(sequence)
    structure, energy = fc.mfe()
    return structure, energy


def compute_bpp_from_samples(structures: List[str]) -> np.ndarray:
    """Estimate base pair probabilities from a list of sampled structures.

    Parameters
    ----------
    structures:
        List of dot-bracket strings of equal length.

    Returns
    -------
    Symmetric (n x n) probability matrix.
    """
    n = len(structures[0])
    counts = np.zeros((n, n))
    for struct in structures:
        for i, j in dot_bracket_to_pairs(struct):
            counts[i, j] += 1
            counts[j, i] += 1
    return counts / len(structures)


def dot_bracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    """Convert a dot-bracket string to a list of (i, j) base-pair tuples (0-indexed)."""
    pairs: List[Tuple[int, int]] = []
    stack: List[int] = []
    for idx, char in enumerate(structure):
        if char == "(":
            stack.append(idx)
        elif char == ")":
            partner = stack.pop()
            pairs.append((partner, idx))
    return pairs


def pairs_to_dot_bracket(pairs: List[Tuple[int, int]], length: int) -> str:
    """Convert a list of (i, j) pairs to a dot-bracket string."""
    db = ["."] * length
    for i, j in pairs:
        db[i] = "("
        db[j] = ")"
    return "".join(db)


def structure_to_pair_table(structure: str) -> np.ndarray:
    """Return a pair-table array: ``pt[i] = j`` if i pairs with j, else -1 (0-indexed)."""
    n = len(structure)
    pt = np.full(n, -1, dtype=int)
    for i, j in dot_bracket_to_pairs(structure):
        pt[i] = j
        pt[j] = i
    return pt
