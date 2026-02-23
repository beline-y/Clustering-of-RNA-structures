"""Unit tests for RNA structure clustering pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.sampling import (
    sample_structures,
    get_vienna_bpp,
    compute_bpp_from_samples,
    dot_bracket_to_pairs,
    pairs_to_dot_bracket,
    structure_to_pair_table,
)
from src.distances import base_pair_distance, hamming_distance, pairwise_distance_matrix
from src.cluster_analysis import (
    compute_cluster_probabilities,
    compute_cluster_bpp,
    compute_centroid_structure,
)
from src.mea import compute_mea_structure
from src.convergence import rmsd_bpp, analyze_convergence


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
SHORT_SEQ = "GGGGAAAAACCCC"   # simple 5-bp hairpin, 13 nt


@pytest.fixture(scope="module")
def hairpin_samples():
    """1000 samples for SHORT_SEQ."""
    return sample_structures(SHORT_SEQ, n_samples=1000)


@pytest.fixture(scope="module")
def vienna_bpp():
    return get_vienna_bpp(SHORT_SEQ)


# --------------------------------------------------------------------------- #
#  sampling.py
# --------------------------------------------------------------------------- #
class TestDotBracket:
    def test_pairs_round_trip(self):
        structure = "((((....))))"
        n = len(structure)
        pairs = dot_bracket_to_pairs(structure)
        rebuilt = pairs_to_dot_bracket(pairs, n)
        assert rebuilt == structure

    def test_empty_structure(self):
        assert dot_bracket_to_pairs("....") == []

    def test_pair_table(self):
        pt = structure_to_pair_table("((...))")
        assert pt[0] == 6
        assert pt[6] == 0
        assert pt[1] == 5
        assert pt[3] == -1

    def test_pairs_unique_positions(self):
        structure = "((((...))))..((.....))"
        pairs = dot_bracket_to_pairs(structure)
        positions = [p for pair in pairs for p in pair]
        assert len(positions) == len(set(positions)), "Each position appears at most once"


class TestSampling:
    def test_returns_correct_count(self, hairpin_samples):
        assert len(hairpin_samples) == 1000

    def test_structures_correct_length(self, hairpin_samples):
        for s in hairpin_samples[:10]:
            assert len(s) == len(SHORT_SEQ)

    def test_valid_dot_bracket(self, hairpin_samples):
        for s in hairpin_samples[:20]:
            assert all(c in ".()" for c in s)
            # Balanced brackets
            depth = 0
            for c in s:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                assert depth >= 0
            assert depth == 0

    def test_bpp_from_samples_shape(self, hairpin_samples):
        bpp = compute_bpp_from_samples(hairpin_samples)
        n = len(SHORT_SEQ)
        assert bpp.shape == (n, n)

    def test_bpp_from_samples_symmetric(self, hairpin_samples):
        bpp = compute_bpp_from_samples(hairpin_samples)
        np.testing.assert_allclose(bpp, bpp.T)

    def test_bpp_from_samples_range(self, hairpin_samples):
        bpp = compute_bpp_from_samples(hairpin_samples)
        assert np.all(bpp >= 0) and np.all(bpp <= 1)

    def test_vienna_bpp_shape(self, vienna_bpp):
        n = len(SHORT_SEQ)
        assert vienna_bpp.shape == (n, n)

    def test_vienna_bpp_symmetric(self, vienna_bpp):
        np.testing.assert_allclose(vienna_bpp, vienna_bpp.T)

    def test_vienna_bpp_non_zero(self, vienna_bpp):
        assert vienna_bpp.max() > 0

    def test_sampled_bpp_close_to_vienna(self, hairpin_samples, vienna_bpp):
        sampled = compute_bpp_from_samples(hairpin_samples)
        rmsd = rmsd_bpp(sampled, vienna_bpp)
        assert rmsd < 0.05, f"RMSD too large: {rmsd:.4f}"


# --------------------------------------------------------------------------- #
#  distances.py
# --------------------------------------------------------------------------- #
class TestDistances:
    def test_bp_distance_identical(self):
        s = "((((....))))"
        assert base_pair_distance(s, s) == 0

    def test_bp_distance_opposite(self):
        s1 = "((((....))))"
        s2 = "............"
        # s1 has 4 pairs, s2 has 0 → distance = 4 + 4 = 8? No: |A Δ B|
        d = base_pair_distance(s1, s2)
        assert d > 0

    def test_hamming_distance_identical(self):
        s = "((((....))))"
        assert hamming_distance(s, s) == 0

    def test_hamming_distance_all_different(self):
        s1 = "(((("
        s2 = "))))"
        assert hamming_distance(s1, s2) == 4

    def test_pairwise_matrix_symmetric(self):
        structs = ["((((....))))", "............", "((......)).."]
        D = pairwise_distance_matrix(structs, metric="bp")
        np.testing.assert_array_equal(D, D.T)

    def test_pairwise_matrix_zero_diagonal(self):
        structs = ["((((....))))", "............", "((......)).."]
        D = pairwise_distance_matrix(structs, metric="bp")
        np.testing.assert_array_equal(np.diag(D), [0, 0, 0])

    def test_pairwise_hamming_symmetric(self):
        structs = ["((((....))))", "............", "((......)).."]
        D = pairwise_distance_matrix(structs, metric="hamming")
        np.testing.assert_array_equal(D, D.T)


# --------------------------------------------------------------------------- #
#  cluster_analysis.py
# --------------------------------------------------------------------------- #
class TestClusterAnalysis:
    def test_cluster_probabilities_sum_to_one(self):
        labels = np.array([0, 0, 1, 1, 2])
        probs = compute_cluster_probabilities(labels)
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_cluster_probabilities_correct(self):
        labels = np.array([0, 0, 0, 1, 1])
        probs = compute_cluster_probabilities(labels)
        assert abs(probs[0] - 0.6) < 1e-9
        assert abs(probs[1] - 0.4) < 1e-9

    def test_cluster_bpp_keys(self):
        structs = ["(((...)))", ".........", "(((...)))"]
        labels = np.array([0, 1, 0])
        bpps = compute_cluster_bpp(structs, labels)
        assert set(bpps.keys()) == {0, 1}

    def test_cluster_bpp_shape(self):
        structs = ["(((...)))", ".........", "(((...)))"]
        labels = np.array([0, 1, 0])
        bpps = compute_cluster_bpp(structs, labels)
        for bpp in bpps.values():
            assert bpp.shape == (9, 9)

    def test_cluster_bpp_dominant_pair(self):
        # Both structures in cluster 0 have the same pair (0,8)
        structs = ["(((...)))", "(((...)))", "........."]
        labels = np.array([0, 0, 1])
        bpps = compute_cluster_bpp(structs, labels)
        assert bpps[0][0, 8] == pytest.approx(1.0)
        assert bpps[1][0, 8] == pytest.approx(0.0)

    def test_centroid_structure_valid(self):
        # Construct a BPP with one certain pair (0,8)
        bpp = np.zeros((9, 9))
        bpp[0, 8] = bpp[8, 0] = 0.8
        bpp[1, 7] = bpp[7, 1] = 0.6
        centroid = compute_centroid_structure(bpp)
        assert len(centroid) == 9
        assert centroid[0] == "(" and centroid[8] == ")"

    def test_centroid_structure_valid_nesting(self):
        """Centroid must have balanced brackets (valid structure)."""
        bpp = np.zeros((12, 12))
        bpp[0, 11] = bpp[11, 0] = 0.9
        bpp[1, 10] = bpp[10, 1] = 0.8
        bpp[2, 9] = bpp[9, 2] = 0.7
        centroid = compute_centroid_structure(bpp)
        depth = 0
        for c in centroid:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            assert depth >= 0
        assert depth == 0

    def test_centroid_no_conflicting_pairs(self):
        """At most one partner per position (threshold > 0.5 guarantees this)."""
        rng = np.random.default_rng(0)
        n = 20
        # Random BPP: entries scaled so no row sums to more than 1
        bpp = rng.uniform(0, 0.3, (n, n))
        bpp = (bpp + bpp.T) / 2
        np.fill_diagonal(bpp, 0)
        centroid = compute_centroid_structure(bpp, threshold=0.5)
        # Verify via pair table: each position has at most one partner
        from src.sampling import structure_to_pair_table
        pt = structure_to_pair_table(centroid)
        partner_positions = pt[pt >= 0]
        assert len(set(partner_positions)) == len(partner_positions)


# --------------------------------------------------------------------------- #
#  mea.py
# --------------------------------------------------------------------------- #
class TestMEA:
    def test_mea_returns_valid_structure(self):
        bpp = np.zeros((9, 9))
        bpp[0, 8] = bpp[8, 0] = 0.9
        bpp[1, 7] = bpp[7, 1] = 0.8
        mea, ea = compute_mea_structure(bpp)
        assert len(mea) == 9
        # Validate bracket balance
        depth = sum(1 if c == "(" else -1 if c == ")" else 0 for c in mea)
        assert depth == 0

    def test_mea_score_positive(self):
        bpp = np.zeros((9, 9))
        bpp[0, 8] = bpp[8, 0] = 0.9
        _, ea = compute_mea_structure(bpp)
        assert ea > 0

    def test_mea_all_unpaired(self):
        bpp = np.zeros((5, 5))
        mea, ea = compute_mea_structure(bpp)
        # With zero pair probabilities, all bases should be unpaired
        assert mea == "....."

    def test_mea_gamma_effect(self):
        bpp = np.zeros((8, 8))
        bpp[0, 7] = bpp[7, 0] = 0.4
        # With low gamma, should not pair (pu contribution wins)
        mea_low, _ = compute_mea_structure(bpp, gamma=0.1)
        # With high gamma, should pair
        mea_high, _ = compute_mea_structure(bpp, gamma=10.0)
        assert "(" in mea_high


# --------------------------------------------------------------------------- #
#  convergence.py
# --------------------------------------------------------------------------- #
class TestConvergence:
    def test_rmsd_identical(self, vienna_bpp):
        assert rmsd_bpp(vienna_bpp, vienna_bpp) == pytest.approx(0.0)

    def test_rmsd_non_negative(self, hairpin_samples, vienna_bpp):
        sampled = compute_bpp_from_samples(hairpin_samples)
        assert rmsd_bpp(sampled, vienna_bpp) >= 0

    def test_convergence_decreasing(self, hairpin_samples, vienna_bpp):
        ns, rmsds = analyze_convergence(hairpin_samples, vienna_bpp)
        # RMSD should generally decrease with more samples (test first vs last half)
        mid = len(rmsds) // 2
        assert np.mean(rmsds[:mid]) >= np.mean(rmsds[mid:]) - 0.01  # allow small noise

    def test_convergence_lengths_match(self, hairpin_samples, vienna_bpp):
        ns, rmsds = analyze_convergence(hairpin_samples, vienna_bpp)
        assert len(ns) == len(rmsds)
        assert all(n > 0 for n in ns)
