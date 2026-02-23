"""Dot-plot and cluster visualizations for RNA base pair probabilities."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_dot_plot(
    bpp_matrix: np.ndarray,
    title: str,
    ax: Optional[plt.Axes] = None,
    min_prob: float = 0.01,
) -> plt.Axes:
    """Draw a single-panel dot plot.

    Each dot at position *(i, j)* has area proportional to the base pair
    probability ``bpp_matrix[i, j]``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    n = bpp_matrix.shape[0]
    rows, cols = np.where(bpp_matrix > min_prob)
    probs = bpp_matrix[rows, cols]
    # Dot area scaled so max probability fills ~1 unit^2
    sizes = probs * 500
    ax.scatter(cols + 1, rows + 1, s=sizes, c=probs, cmap="Blues", vmin=0, vmax=1, alpha=0.8)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    ax.set_xlabel("Position j")
    ax.set_ylabel("Position i")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return ax


def plot_comparison_dot_plot(
    sampled_bpp: np.ndarray,
    vienna_bpp: np.ndarray,
    rna_name: str,
    output_path: Optional[str] = None,
    min_prob: float = 0.01,
) -> plt.Figure:
    """Two-panel dot plot comparing sampled BPPs (left) with Vienna BPPs (right).

    Upper-left panel: BPPs estimated from stochastic samples.
    Upper-right panel: BPPs from the Vienna RNA partition function.
    """
    n = sampled_bpp.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, bpp, subtitle in zip(
        axes, [sampled_bpp, vienna_bpp], ["Sampled BPPs", "Vienna RNA BPPs"]
    ):
        rows, cols = np.where((bpp > min_prob) & (np.triu(np.ones((n, n)), k=1) > 0))
        probs = bpp[rows, cols]
        sc = ax.scatter(
            cols + 1, rows + 1, s=probs * 500, c=probs,
            cmap="Blues", vmin=0, vmax=1, alpha=0.8,
        )
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, n + 1)
        ax.set_xlabel("Position j")
        ax.set_ylabel("Position i")
        ax.set_title(subtitle)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, label="Probability")

    fig.suptitle(f"Dot plot – {rna_name}", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_combined_dot_plot(
    sampled_bpp: np.ndarray,
    vienna_bpp: np.ndarray,
    rna_name: str,
    output_path: Optional[str] = None,
    min_prob: float = 0.01,
) -> plt.Figure:
    """Single-panel dot plot with sampled BPPs in the upper triangle and
    Vienna BPPs in the lower triangle (mirrored), similar to the Vienna RNA
    PostScript dot-plot format.
    """
    n = sampled_bpp.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7))

    # Upper triangle: sampled
    rows, cols = np.where(
        (sampled_bpp > min_prob) & (np.triu(np.ones((n, n)), k=1) > 0)
    )
    if len(rows):
        probs = sampled_bpp[rows, cols]
        ax.scatter(
            cols + 1, n - rows, s=probs * 400, c=probs,
            cmap="Blues", vmin=0, vmax=1, alpha=0.8, label="Sampled",
        )

    # Lower triangle: Vienna (reflected about the diagonal)
    rows2, cols2 = np.where(
        (vienna_bpp > min_prob) & (np.triu(np.ones((n, n)), k=1) > 0)
    )
    if len(rows2):
        probs2 = vienna_bpp[rows2, cols2]
        ax.scatter(
            rows2 + 1, n - cols2, s=probs2 * 400, c=probs2,
            cmap="Reds", vmin=0, vmax=1, alpha=0.8, label="Vienna",
        )

    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    ax.set_title(f"Dot plot – {rna_name}\n(blue=sampled upper, red=Vienna lower)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_2d_embedding(
    coords: np.ndarray,
    labels: np.ndarray,
    rna_name: str,
    method: str = "MDS",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of 2-D embedding coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=20, color=cmap(idx), alpha=0.7, label=f"Cluster {lbl}",
        )
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"{method} embedding – {rna_name}")
    ax.legend(markerscale=2)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cluster_dot_plots(
    cluster_bpps: dict,
    cluster_probs: dict,
    rna_name: str,
    output_path: Optional[str] = None,
    min_prob: float = 0.01,
    max_clusters: int = 6,
) -> plt.Figure:
    """Grid of dot plots, one per cluster, sized by cluster probability."""
    labels = sorted(cluster_bpps.keys())[: max_clusters]
    ncols = min(3, len(labels))
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, lbl in enumerate(labels):
        ax = axes[ax_idx]
        bpp = cluster_bpps[lbl]
        n = bpp.shape[0]
        rows, cols = np.where(
            (bpp > min_prob) & (np.triu(np.ones((n, n)), k=1) > 0)
        )
        if len(rows):
            probs = bpp[rows, cols]
            sc = ax.scatter(
                cols + 1, rows + 1, s=probs * 400, c=probs,
                cmap="Blues", vmin=0, vmax=1, alpha=0.8,
            )
            plt.colorbar(sc, ax=ax, label="P(bp | cluster)")
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, n + 1)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        prob_str = f"{cluster_probs.get(lbl, 0):.2%}"
        ax.set_title(f"Cluster {lbl}  (prob={prob_str})")
        ax.set_xlabel("j")
        ax.set_ylabel("i")

    for ax in axes[len(labels):]:
        ax.set_visible(False)

    fig.suptitle(f"Per-cluster dot plots – {rna_name}", fontsize=13)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence(
    n_samples_list: List[int],
    rmsd_list: List[float],
    rna_name: str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Line plot of BPP RMSD vs number of samples (convergence analysis)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_samples_list, rmsd_list, marker="o", linewidth=2)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("RMSD vs Vienna BPP")
    ax.set_title(f"Convergence of sampled BPPs – {rna_name}")
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
