import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _savefig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_true_vs_inferred_angles(
    output_path: str,
    true_angles: np.ndarray,
    inferred_angles: np.ndarray,
    dimension: int,
) -> None:
    # One figure per requirement: true vs inferred scatter(s).
    if dimension <= 1:
        plt.figure(figsize=(6, 5))
        plt.scatter(true_angles[:, 0], inferred_angles[:, 0], s=10, alpha=0.7)
        line = np.linspace(0.0, 2.0 * np.pi, 200)
        plt.plot(line, line, "k--", linewidth=1)
        plt.xlabel("True theta_0")
        plt.ylabel("Inferred theta_0")
        plt.title("True vs inferred theta_0")
        _savefig(output_path)
        return

    ncols = min(2, true_angles.shape[1])
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    for i in range(ncols):
        ax = axes[i]
        ax.scatter(true_angles[:, i], inferred_angles[:, i], s=10, alpha=0.7)
        lo = min(true_angles[:, i].min(), inferred_angles[:, i].min())
        hi = max(true_angles[:, i].max(), inferred_angles[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel(f"True angle_{i}")
        ax.set_ylabel(f"Inferred angle_{i}")
        ax.set_title(f"True vs inferred angle_{i}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_angular_error_hist(output_path: str, angular_errors: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))
    plt.hist(np.abs(angular_errors), bins=40, edgecolor="black", linewidth=0.5)
    plt.xlabel("Absolute angular/geodesic error")
    plt.ylabel("Count")
    plt.title("Angular Error Distribution")
    _savefig(output_path)


def plot_pairwise_distance_scatter(
    output_path: str,
    true_pairwise: np.ndarray,
    inferred_pairwise: np.ndarray,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(true_pairwise, inferred_pairwise, s=8, alpha=0.5)
    lo = min(np.min(true_pairwise), np.min(inferred_pairwise))
    hi = max(np.max(true_pairwise), np.max(inferred_pairwise))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("True pairwise distance")
    plt.ylabel("Inferred pairwise distance")
    plt.title("Pairwise Distance: True vs Inferred")
    _savefig(output_path)


def plot_polar_d1(
    output_path: str,
    true_theta: np.ndarray,
    inferred_theta: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    r_true = np.ones_like(true_theta)
    r_inf = np.full_like(inferred_theta, 0.92)
    ax.scatter(true_theta, r_true, s=10, alpha=0.7, label="true")
    ax.scatter(inferred_theta, r_inf, s=10, alpha=0.7, label="inferred")
    ax.set_rticks([])
    ax.set_title("S1 positions: true vs inferred")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_projected_d2(
    output_path: str,
    true_vectors: np.ndarray,
    inferred_vectors: np.ndarray,
) -> None:
    # 2D projection of points on S^2 represented in R^3 (or higher by taking first two axes).
    plt.figure(figsize=(6, 5))
    plt.scatter(true_vectors[:, 0], true_vectors[:, 1], s=10, alpha=0.6, label="true")
    plt.scatter(inferred_vectors[:, 0], inferred_vectors[:, 1], s=10, alpha=0.6, label="inferred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Projected S^D Coordinates (first two components)")
    plt.legend()
    _savefig(output_path)


def save_all_required_plots(
    output_dir: str,
    dimension: int,
    true_angles: np.ndarray,
    inferred_angles: np.ndarray,
    angular_errors: np.ndarray,
    true_pairwise: np.ndarray,
    inferred_pairwise: np.ndarray,
    true_vectors: Optional[np.ndarray] = None,
    inferred_vectors: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    plot_true_vs_inferred_angles(
        os.path.join(output_dir, "scatter_true_vs_inferred_angles.png"),
        true_angles,
        inferred_angles,
        dimension,
    )
    plot_angular_error_hist(
        os.path.join(output_dir, "hist_angular_errors.png"),
        angular_errors,
    )
    plot_pairwise_distance_scatter(
        os.path.join(output_dir, "scatter_pairwise_distances.png"),
        true_pairwise,
        inferred_pairwise,
    )

    if dimension == 1:
        plot_polar_d1(
            os.path.join(output_dir, "polar_true_vs_inferred.png"),
            true_angles[:, 0],
            inferred_angles[:, 0],
        )

    if dimension == 2 and true_vectors is not None and inferred_vectors is not None:
        plot_projected_d2(
            os.path.join(output_dir, "projected_d2_points.png"),
            true_vectors,
            inferred_vectors,
        )
