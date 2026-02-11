#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    print(
        "ERROR: Missing dependency for validation pipeline. "
        "Install numpy, pandas, and matplotlib.",
        file=sys.stderr,
    )
    print(f"DETAILS: {exc}", file=sys.stderr)
    sys.exit(2)

from plot_sd import save_all_required_plots


def run_command(cmd, cwd: Path) -> None:
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(str(c) for c in cmd)
            + "\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )


def sample_kappas(
    n: int,
    gamma: float,
    target_avg_degree: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if gamma <= 2.0:
        raise ValueError("gamma must be > 2")
    # Pareto tail with expected value close to target_avg_degree.
    kappa_min = target_avg_degree * (gamma - 2.0) / (gamma - 1.0)
    return kappa_min * (1.0 + rng.pareto(gamma - 1.0, size=n))


def random_unit_vectors(n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(n, dimension + 1))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return raw / norms


def write_hidden_variables_file(
    path: Path,
    kappas: np.ndarray,
    dimension: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    with path.open("w", encoding="utf-8") as f:
        if dimension == 1:
            theta = rng.uniform(0.0, 2.0 * np.pi, size=len(kappas))
            for k, t in zip(kappas, theta):
                f.write(f"{k:.17g} {t:.17g}\n")
        else:
            vectors = random_unit_vectors(len(kappas), dimension, rng)
            for i, k in enumerate(kappas):
                coords = " ".join(f"{x:.17g}" for x in vectors[i])
                f.write(f"{k:.17g} {coords}\n")


def load_standard_csv(path: Path, dimension: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["node_id", "r", "kappa"]
    if dimension == 1:
        expected += ["theta_0"]
    else:
        expected += [f"pos_{i}" for i in range(dimension + 1)]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df[expected].sort_values("node_id").reset_index(drop=True)


def load_legacy_inf_coord(path: Path, dimension: int) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if dimension == 1:
                if len(parts) < 4:
                    continue
                rows.append(
                    {
                        "node_id": parts[0],
                        "r": float(parts[3]),
                        "kappa": float(parts[1]),
                        "theta_0": float(parts[2]),
                    }
                )
            else:
                expected = 3 + (dimension + 1)
                if len(parts) < expected:
                    continue
                row = {
                    "node_id": parts[0],
                    "r": float(parts[2]),
                    "kappa": float(parts[1]),
                }
                for i in range(dimension + 1):
                    row[f"pos_{i}"] = float(parts[3 + i])
                rows.append(row)

    if not rows:
        raise ValueError(f"No coordinate rows parsed from {path}")
    return pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def vectors_to_hyperspherical_angles(vectors: np.ndarray) -> np.ndarray:
    # Converts each R^(D+1) unit vector to D angular coordinates.
    vectors = normalize_rows(vectors)
    d_plus_1 = vectors.shape[1]
    d = d_plus_1 - 1
    out = np.zeros((vectors.shape[0], d), dtype=float)

    for idx, vec in enumerate(vectors):
        for i in range(d - 1):
            bottom = np.linalg.norm(vec[i:])
            ratio = 1.0 if bottom == 0.0 else vec[i] / bottom
            ratio = np.clip(ratio, -1.0, 1.0)
            out[idx, i] = math.acos(ratio)
        out[idx, d - 1] = np.mod(np.arctan2(vec[-1], vec[-2]), 2.0 * np.pi)

    return out


def circular_signed_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def align_angles_to_reference(
    reference: np.ndarray,
    target: np.ndarray,
    n_shifts: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    shifts = np.linspace(0.0, 2.0 * np.pi, num=n_shifts, endpoint=False)

    def evaluate(candidate: np.ndarray):
        best_mse = float("inf")
        best_shift = 0.0
        best_aligned = candidate
        best_err = circular_signed_diff(candidate, reference)
        for s in shifts:
            aligned = np.mod(candidate + s, 2.0 * np.pi)
            err = circular_signed_diff(aligned, reference)
            mse = float(np.mean(err * err))
            if mse < best_mse:
                best_mse = mse
                best_shift = float(s)
                best_aligned = aligned
                best_err = err
        return best_aligned, best_err, best_mse, best_shift

    direct_aligned, direct_err, direct_mse, direct_shift = evaluate(np.mod(target, 2.0 * np.pi))
    reflected_aligned, reflected_err, reflected_mse, reflected_shift = evaluate(np.mod(-target, 2.0 * np.pi))

    if reflected_mse < direct_mse:
        return reflected_aligned, reflected_err, {
            "reflection": 1.0,
            "shift": reflected_shift,
            "mse": reflected_mse,
        }
    return direct_aligned, direct_err, {
        "reflection": 0.0,
        "shift": direct_shift,
        "mse": direct_mse,
    }


def procrustes_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    reference = normalize_rows(reference)
    target = normalize_rows(target)
    m = target.T @ reference
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    aligned = target @ r
    return normalize_rows(aligned)


def sample_pairs(n: int, num_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    i = rng.integers(0, n, size=num_pairs * 3)
    j = rng.integers(0, n, size=num_pairs * 3)
    mask = i != j
    i = i[mask][:num_pairs]
    j = j[mask][:num_pairs]
    if len(i) < num_pairs:
        extra = num_pairs - len(i)
        i2, j2 = sample_pairs(n, extra, rng)
        i = np.concatenate([i, i2])
        j = np.concatenate([j, j2])
    return i, j


def pairwise_distances_from_angles(theta: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.pi - np.abs(np.pi - np.abs(theta[i] - theta[j]))


def pairwise_distances_from_vectors(vectors: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    dots = np.sum(vectors[i] * vectors[j], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(rx, ry)


def compute_error_metrics(errors: np.ndarray, dist_true: np.ndarray, dist_inf: np.ndarray) -> Dict[str, float]:
    return {
        "mean_error": float(np.mean(np.abs(errors))),
        "median_error": float(np.median(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "pearson_pairwise": pearson_corr(dist_true, dist_inf),
        "spearman_pairwise": spearman_corr(dist_true, dist_inf),
    }


def evaluate_comparison(
    reference_repr: np.ndarray,
    target_repr: np.ndarray,
    dimension: int,
    num_pair_samples: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    n = reference_repr.shape[0]
    i, j = sample_pairs(n, num_pair_samples, rng)

    if dimension == 1:
        aligned, errors, alignment = align_angles_to_reference(reference_repr, target_repr)
        dist_ref = pairwise_distances_from_angles(reference_repr, i, j)
        dist_target = pairwise_distances_from_angles(aligned, i, j)
        metrics = compute_error_metrics(errors, dist_ref, dist_target)
        metrics.update({f"alignment_{k}": float(v) for k, v in alignment.items()})
        payload = {
            "reference_angles": reference_repr.reshape(-1, 1),
            "aligned_target_angles": aligned.reshape(-1, 1),
            "errors": errors,
            "pairwise_reference": dist_ref,
            "pairwise_target": dist_target,
        }
        return metrics, payload

    ref_vectors = normalize_rows(reference_repr)
    target_vectors = normalize_rows(target_repr)
    aligned_target = procrustes_align(ref_vectors, target_vectors)

    dots = np.sum(ref_vectors * aligned_target, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    errors = np.arccos(dots)

    dist_ref = pairwise_distances_from_vectors(ref_vectors, i, j)
    dist_target = pairwise_distances_from_vectors(aligned_target, i, j)
    metrics = compute_error_metrics(errors, dist_ref, dist_target)

    payload = {
        "reference_vectors": ref_vectors,
        "aligned_target_vectors": aligned_target,
        "reference_angles": vectors_to_hyperspherical_angles(ref_vectors),
        "aligned_target_angles": vectors_to_hyperspherical_angles(aligned_target),
        "errors": errors,
        "pairwise_reference": dist_ref,
        "pairwise_target": dist_target,
    }
    return metrics, payload


def prepare_common_node_order(
    truth_df: pd.DataFrame,
    original_df: pd.DataFrame,
    refactored_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = (
        set(truth_df["node_id"]) & set(original_df["node_id"]) & set(refactored_df["node_id"])
    )
    if not common:
        raise RuntimeError("No common node ids across truth/original/refactored outputs")

    common_sorted = sorted(common)

    def keep(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[df["node_id"].isin(common_sorted)]
            .sort_values("node_id")
            .reset_index(drop=True)
        )

    return keep(truth_df), keep(original_df), keep(refactored_df)


def extract_representation(df: pd.DataFrame, dimension: int) -> np.ndarray:
    if dimension == 1:
        return df["theta_0"].to_numpy(dtype=float)
    return df[[f"pos_{i}" for i in range(dimension + 1)]].to_numpy(dtype=float)


def default_beta_for_dimension(dimension: int) -> float:
    return float(dimension * 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate refactored D-Mercator embedding against legacy and ground truth.")
    parser.add_argument("--build-dir", default="build", help="Directory containing compiled executables")
    parser.add_argument("--output-dir", default="validation_output", help="Directory for generated artifacts")
    parser.add_argument("--dimension", type=int, default=1, help="S^D dimension")
    parser.add_argument("--nodes", type=int, default=500, help="Number of synthetic nodes")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed")
    parser.add_argument("--gamma", type=float, default=2.7, help="Power-law exponent for kappas")
    parser.add_argument("--target-avg-degree", type=float, default=10.0, help="Target average kappa")
    parser.add_argument("--beta", type=float, default=None, help="Beta parameter for generation/embedding")
    parser.add_argument("--pair-samples", type=int, default=5000, help="Number of node pairs for distance correlation")
    parser.add_argument("--assert-rmse-ref-orig", type=float, default=1e-6, help="Max RMSE for refactored vs original")
    parser.add_argument("--assert-rmse-ref-truth", type=float, default=1.6, help="Max RMSE for refactored vs truth")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (repo_root / build_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    beta = args.beta if args.beta is not None else default_beta_for_dimension(args.dimension)

    generate_bin = build_dir / "generate_sd"
    legacy_bin = build_dir / "mercator_legacy"
    refactored_bin = build_dir / "embed_sd"

    for exe in (generate_bin, legacy_bin, refactored_bin):
        if not exe.exists():
            raise FileNotFoundError(f"Missing executable: {exe}")

    rng = np.random.default_rng(args.seed)

    hidden_file = output_dir / "hidden_variables.txt"
    generate_root = output_dir / "synthetic_sd"
    original_root = output_dir / "original"
    refactored_root = output_dir / "refactored"

    kappas = sample_kappas(args.nodes, args.gamma, args.target_avg_degree, rng)
    write_hidden_variables_file(hidden_file, kappas, args.dimension, args.seed + 17)

    run_command(
        [
            generate_bin,
            "-d",
            str(args.dimension),
            "-b",
            str(beta),
            "-s",
            str(args.seed),
            "-o",
            str(generate_root),
            "-t",
            str(hidden_file),
        ],
        cwd=repo_root,
    )

    edge_file = Path(str(generate_root) + ".edge")
    truth_csv = Path(str(generate_root) + ".truth.csv")

    run_command(
        [
            legacy_bin,
            "-q",
            "-d",
            str(args.dimension),
            "-s",
            str(args.seed),
            "-b",
            str(beta),
            "-o",
            str(original_root),
            str(edge_file),
        ],
        cwd=repo_root,
    )

    run_command(
        [
            refactored_bin,
            "-q",
            "-d",
            str(args.dimension),
            "-s",
            str(args.seed),
            "-b",
            str(beta),
            "-o",
            str(refactored_root),
            str(edge_file),
        ],
        cwd=repo_root,
    )

    original_inf = Path(str(original_root) + ".inf_coord")
    refactored_csv = Path(str(refactored_root) + ".coords.csv")

    truth_df = load_standard_csv(truth_csv, args.dimension)
    original_df = load_legacy_inf_coord(original_inf, args.dimension)
    refactored_df = load_standard_csv(refactored_csv, args.dimension)

    truth_df, original_df, refactored_df = prepare_common_node_order(
        truth_df,
        original_df,
        refactored_df,
    )

    coords_truth_csv = output_dir / "coords_truth.csv"
    coords_original_csv = output_dir / "coords_original.csv"
    coords_refactored_csv = output_dir / "coords_refactored.csv"

    truth_df.to_csv(coords_truth_csv, index=False)
    original_df.to_csv(coords_original_csv, index=False)
    refactored_df.to_csv(coords_refactored_csv, index=False)

    truth_repr = extract_representation(truth_df, args.dimension)
    original_repr = extract_representation(original_df, args.dimension)
    refactored_repr = extract_representation(refactored_df, args.dimension)

    metrics_truth_original, _ = evaluate_comparison(
        truth_repr,
        original_repr,
        args.dimension,
        args.pair_samples,
        np.random.default_rng(args.seed + 101),
    )
    metrics_truth_refactored, payload_truth_ref = evaluate_comparison(
        truth_repr,
        refactored_repr,
        args.dimension,
        args.pair_samples,
        np.random.default_rng(args.seed + 202),
    )
    metrics_original_refactored, _ = evaluate_comparison(
        original_repr,
        refactored_repr,
        args.dimension,
        args.pair_samples,
        np.random.default_rng(args.seed + 303),
    )

    plots_dir = output_dir / "plots"
    if args.dimension == 1:
        save_all_required_plots(
            str(plots_dir),
            args.dimension,
            payload_truth_ref["reference_angles"],
            payload_truth_ref["aligned_target_angles"],
            payload_truth_ref["errors"],
            payload_truth_ref["pairwise_reference"],
            payload_truth_ref["pairwise_target"],
        )
    else:
        save_all_required_plots(
            str(plots_dir),
            args.dimension,
            payload_truth_ref["reference_angles"],
            payload_truth_ref["aligned_target_angles"],
            payload_truth_ref["errors"],
            payload_truth_ref["pairwise_reference"],
            payload_truth_ref["pairwise_target"],
            true_vectors=payload_truth_ref["reference_vectors"],
            inferred_vectors=payload_truth_ref["aligned_target_vectors"],
        )

    metrics = {
        "truth_vs_original": metrics_truth_original,
        "truth_vs_refactored": metrics_truth_refactored,
        "original_vs_refactored": metrics_original_refactored,
    }

    summary = {
        "config": {
            "dimension": args.dimension,
            "nodes": args.nodes,
            "seed": args.seed,
            "gamma": args.gamma,
            "target_avg_degree": args.target_avg_degree,
            "beta": beta,
            "pair_samples": args.pair_samples,
        },
        "node_count": int(len(truth_df)),
        "paths": {
            "edge_file": str(edge_file),
            "truth_csv": str(coords_truth_csv),
            "original_csv": str(coords_original_csv),
            "refactored_csv": str(coords_refactored_csv),
            "plots_dir": str(plots_dir),
        },
        "metrics": metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Validation summary")
    print(f"  Dimension: {args.dimension}")
    print(f"  Nodes compared: {len(truth_df)}")
    print("  Metrics:")
    print(f"    truth_vs_original.rmse      = {metrics_truth_original['rmse']:.8f}")
    print(f"    truth_vs_refactored.rmse    = {metrics_truth_refactored['rmse']:.8f}")
    print(f"    original_vs_refactored.rmse = {metrics_original_refactored['rmse']:.8f}")
    print(f"    truth_vs_refactored.pearson_pairwise  = {metrics_truth_refactored['pearson_pairwise']:.8f}")
    print(f"    truth_vs_refactored.spearman_pairwise = {metrics_truth_refactored['spearman_pairwise']:.8f}")
    print(f"  Plots: {plots_dir}")
    print(f"  Metrics JSON: {metrics_path}")

    if metrics_original_refactored["rmse"] > args.assert_rmse_ref_orig:
        raise AssertionError(
            "Refactored-vs-original RMSE exceeded threshold: "
            f"{metrics_original_refactored['rmse']:.8f} > {args.assert_rmse_ref_orig}"
        )

    if metrics_truth_refactored["rmse"] > args.assert_rmse_ref_truth:
        raise AssertionError(
            "Refactored-vs-truth RMSE exceeded threshold: "
            f"{metrics_truth_refactored['rmse']:.8f} > {args.assert_rmse_ref_truth}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
