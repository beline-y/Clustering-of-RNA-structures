# Clustering of RNA Structures

Boltzmann sampling, base pair probability visualisation, and clustering of
RNA secondary structure ensembles using the [Vienna RNA package](https://www.tbi.univie.ac.at/RNA/).

---

## Overview

This project implements the full pipeline described in the course project
requirements:

| Step | Module | Description |
|------|--------|-------------|
| Boltzmann sampling | `src/sampling.py` | Generate structures with `fc.pbacktrack()` |
| Base pair probabilities | `src/sampling.py` | Sampled BPPs + Vienna partition-function BPPs |
| Dot plots | `src/visualization.py` | Two-panel and combined dot-plot figures |
| Structural distances | `src/distances.py` | Base-pair distance and Hamming distance |
| Dimensionality reduction | `src/clustering.py` | MDS and t-SNE on pairwise distance matrix |
| Clustering | `src/clustering.py` | k-means; optimal k via silhouette score |
| Cluster analysis | `src/cluster_analysis.py` | Per-cluster BPPs, cluster probabilities |
| Centroid structures | `src/cluster_analysis.py` | All base pairs with conditional prob > 50 % |
| MEA structures | `src/mea.py` | Nussinov-type DP to maximise expected accuracy |
| Convergence | `src/convergence.py` | RMSD vs Vienna BPPs as function of sample size |

---

## RNA Sequences

Four sequences are analysed (defined in `src/sequences.py`):

| ID | Description | Length | Type |
|----|-------------|--------|------|
| `tRNA_Phe` | Yeast phenylalanine tRNA | 76 nt | Natural |
| `5S_rRNA` | *E. coli* 5S rRNA fragment | 84 nt | Natural |
| `hammerhead` | Hammerhead ribozyme minimal motif | 58 nt | Natural |
| `bistable` | Artificial bistable with two competing CGCGCG hairpins | 28 nt | Artificial |

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

> The `ViennaRNA` package requires Python ≥ 3.8.

### Run the full pipeline

```bash
python main.py                   # all sequences
python main.py bistable          # single sequence
python main.py tRNA_Phe 5S_rRNA  # multiple sequences
```

Results are saved to `results/<rna_name>/`:

| File | Content |
|------|---------|
| `dot_plot_comparison.png` | Side-by-side sampled vs Vienna BPPs |
| `dot_plot_combined.png` | Single-panel dot plot (sampled upper / Vienna lower triangle) |
| `mds_clusters.png` | 2-D MDS embedding coloured by cluster |
| `tsne_clusters.png` | 2-D t-SNE embedding coloured by cluster |
| `elbow_plot.png` | Inertia and silhouette score vs k |
| `cluster_dot_plots.png` | Per-cluster BPP dot plots |
| `convergence.png` | BPP RMSD vs number of samples |
| `summary.txt` | Centroid and MEA structure strings + statistics |

---

## Methods

### Boltzmann sampling

The Vienna RNA fold compound is created with `md.uniq_ML = 1` (unique
multiloop decomposition), which is required to enable stochastic traceback.
After computing the partition function with `fc.pf()`, structures are drawn
from the Boltzmann distribution with `fc.pbacktrack(n_samples)`.

### Base pair probabilities

Sampled BPPs are estimated as the fraction of structures containing each
base pair.  Reference BPPs are taken from the Vienna RNA partition function
(`fc.bpp()`).  Convergence is assessed by the RMSD between the two as a
function of sample count.

### Structural distances

Base-pair distance (symmetric set difference of base-pair sets, via
`RNA.bp_distance`) is used as the default structural metric.  Hamming
distance of dot-bracket strings is also available.

### Clustering

The (1000 × 1000) pairwise distance matrix is embedded in 2-D using
Multidimensional Scaling (MDS) and t-SNE.  k-means clustering is applied to
the MDS coordinates; the optimal number of clusters is selected by the
highest silhouette score over k = 2 … 8.

### Centroid structures

For each cluster *c*, the centroid structure includes all base pairs (i, j)
whose conditional probability P(i, j | c) exceeds 0.5.  This always yields a
valid nested structure: if two positions j ≠ k both had conditional
probability > 0.5 to pair with position i, their sum would exceed 1—a
contradiction since at most one partner can be paired at any position.

### MEA structures

The Maximum Expected Accuracy (MEA) structure maximises

    EA(R) = Σ_{(i,j)∈R} 2·γ·p_{ij}  +  Σ_{i unpaired in R} pu_i

via a Nussinov-type dynamic programming algorithm (O(n³) time).

---

## Key Findings

- **Bistable RNA**: two clusters at ~70 % / ~30 % corresponding to the two
  competing hairpins; centroid structures recover the expected conformations
  exactly.
- **tRNA-Phe**: ~44 % / ~56 % split; clusters correspond to the canonical
  cloverleaf structure vs. alternative pairings.
- **5S rRNA**: three clusters at ~13 % / ~56 % / ~31 %; minor cluster shows
  long-range alternative pairing absent in the MFE structure.
- **Convergence**: RMSD to the Vienna partition-function BPPs drops below
  0.002 by ~1000 samples for all sequences tested.  Longer sequences
  (>80 nt) require more samples to achieve the same RMSD, consistent with a
  larger structural space.
