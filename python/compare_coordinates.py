#!/usr/bin/env python3

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TWO_PI = 2.0 * math.pi


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % TWO_PI - math.pi


@dataclass
class AlignmentResult:
    aligned_df: pd.DataFrame
    transform: Dict[str, float]
    angular_errors: np.ndarray


def load_coordinates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "node_id" not in df.columns:
        raise ValueError(f"Missing node_id column in {path}")
    return df


def best_global_rotation(theta_ref: np.ndarray, theta_mov: np.ndarray) -> float:
    delta = np.angle(np.sum(np.exp(1j * (theta_ref - theta_mov))))
    return float(delta)


def align_dimension_one(ref_df: pd.DataFrame, mov_df: pd.DataFrame) -> AlignmentResult:
    merged = ref_df.merge(mov_df, on="node_id", suffixes=("_ref", "_mov"), how="inner")
    if merged.empty:
        raise ValueError("No common node_id entries found between reference and moving coordinates")

    theta_ref = merged["theta_0_ref"].to_numpy(dtype=float)
    theta_mov = merged["theta_0_mov"].to_numpy(dtype=float)

    delta_plain = best_global_rotation(theta_ref, theta_mov)
    theta_plain = wrap_angle(theta_mov + delta_plain)
    err_plain = wrap_angle(theta_plain - theta_ref)
    rmse_plain = float(np.sqrt(np.mean(err_plain * err_plain)))

    theta_reflected = -theta_mov
    delta_reflect = best_global_rotation(theta_ref, theta_reflected)
    theta_reflect_aligned = wrap_angle(theta_reflected + delta_reflect)
    err_reflect = wrap_angle(theta_reflect_aligned - theta_ref)
    rmse_reflect = float(np.sqrt(np.mean(err_reflect * err_reflect)))

    use_reflection = rmse_reflect < rmse_plain
    if use_reflection:
        theta_aligned = theta_reflect_aligned
        theta_err = err_reflect
        transform = {"reflection": 1.0, "rotation_delta": delta_reflect}
    else:
        theta_aligned = theta_plain
        theta_err = err_plain
        transform = {"reflection": 0.0, "rotation_delta": delta_plain}

    aligned = pd.DataFrame({
        "node_id": merged["node_id"],
        "kappa_ref": merged["kappa_ref"],
        "kappa_mov": merged["kappa_mov"],
        "r_ref": merged["r_ref"],
        "r_mov": merged["r_mov"],
        "theta_0_ref": theta_ref,
        "theta_0_mov_aligned": theta_aligned,
        "theta_0_error": theta_err,
    })

    return AlignmentResult(aligned_df=aligned, transform=transform, angular_errors=np.abs(theta_err))


def orthogonal_procrustes(mov: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a = mov.T @ ref
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    return u @ vt


def align_dimension_gt_one(ref_df: pd.DataFrame, mov_df: pd.DataFrame, dimension: int) -> AlignmentResult:
    merged = ref_df.merge(mov_df, on="node_id", suffixes=("_ref", "_mov"), how="inner")
    if merged.empty:
        raise ValueError("No common node_id entries found between reference and moving coordinates")

    pos_fields = [f"pos_{i}" for i in range(dimension + 1)]
    x_ref = merged[[f"{c}_ref" for c in pos_fields]].to_numpy(dtype=float)
    x_mov = merged[[f"{c}_mov" for c in pos_fields]].to_numpy(dtype=float)

    rot = orthogonal_procrustes(x_mov, x_ref)
    x_mov_aligned = x_mov @ rot

    ref_norm = np.linalg.norm(x_ref, axis=1, keepdims=True)
    mov_norm = np.linalg.norm(x_mov_aligned, axis=1, keepdims=True)
    ref_unit = x_ref / np.clip(ref_norm, 1e-16, None)
    mov_unit = x_mov_aligned / np.clip(mov_norm, 1e-16, None)
    dots = np.sum(ref_unit * mov_unit, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang_err = np.arccos(dots)

    aligned = pd.DataFrame({
        "node_id": merged["node_id"],
        "kappa_ref": merged["kappa_ref"],
        "kappa_mov": merged["kappa_mov"],
        "r_ref": merged["r_ref"],
        "r_mov": merged["r_mov"],
    })
    for i, c in enumerate(pos_fields):
        aligned[f"{c}_ref"] = x_ref[:, i]
        aligned[f"{c}_mov_aligned"] = x_mov_aligned[:, i]
        aligned[f"{c}_error"] = x_mov_aligned[:, i] - x_ref[:, i]

    return AlignmentResult(
        aligned_df=aligned,
        transform={"procrustes_det": float(np.linalg.det(rot))},
        angular_errors=ang_err,
    )


def sample_pair_indices(n: int, sample_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least two nodes to sample pairwise distances")
    i = rng.integers(0, n, size=sample_size)
    j = rng.integers(0, n - 1, size=sample_size)
    j = np.where(j >= i, j + 1, j)
    mask = i < j
    i2 = np.where(mask, i, j)
    j2 = np.where(mask, j, i)
    return i2, j2


def pairwise_distance_correlation(aligned: AlignmentResult,
                                  dimension: int,
                                  pair_samples: int,
                                  seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(aligned.aligned_df)
    i, j = sample_pair_indices(n, pair_samples, rng)

    if dimension == 1:
        t_ref = aligned.aligned_df["theta_0_ref"].to_numpy(dtype=float)
        t_mov = aligned.aligned_df["theta_0_mov_aligned"].to_numpy(dtype=float)
        x_ref = np.column_stack((np.cos(t_ref), np.sin(t_ref)))
        x_mov = np.column_stack((np.cos(t_mov), np.sin(t_mov)))
    else:
        pos_fields = [f"pos_{k}" for k in range(dimension + 1)]
        x_ref = aligned.aligned_df[[f"{c}_ref" for c in pos_fields]].to_numpy(dtype=float)
        x_mov = aligned.aligned_df[[f"{c}_mov_aligned" for c in pos_fields]].to_numpy(dtype=float)

    d_ref = np.linalg.norm(x_ref[i] - x_ref[j], axis=1)
    d_mov = np.linalg.norm(x_mov[i] - x_mov[j], axis=1)

    pearson = float(np.corrcoef(d_ref, d_mov)[0, 1])
    rank_ref = pd.Series(d_ref).rank(method="average").to_numpy(dtype=float)
    rank_mov = pd.Series(d_mov).rank(method="average").to_numpy(dtype=float)
    spearman = float(np.corrcoef(rank_ref, rank_mov)[0, 1])

    return {
        "pair_samples": int(pair_samples),
        "pearson": pearson,
        "spearman": spearman,
    }


def compute_field_metrics(aligned: AlignmentResult, dimension: int) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    for field in ("kappa", "r"):
        diff = aligned.aligned_df[f"{field}_mov"] - aligned.aligned_df[f"{field}_ref"]
        arr = diff.to_numpy(dtype=float)
        metrics[field] = {
            "rmse": float(np.sqrt(np.mean(arr * arr))),
            "max_abs": float(np.max(np.abs(arr))),
            "mean_abs": float(np.mean(np.abs(arr))),
        }

    if dimension == 1:
        arr = aligned.aligned_df["theta_0_error"].to_numpy(dtype=float)
        metrics["theta_0"] = {
            "rmse": float(np.sqrt(np.mean(arr * arr))),
            "max_abs": float(np.max(np.abs(arr))),
            "mean_abs": float(np.mean(np.abs(arr))),
        }
    else:
        for i in range(dimension + 1):
            arr = aligned.aligned_df[f"pos_{i}_error"].to_numpy(dtype=float)
            metrics[f"pos_{i}"] = {
                "rmse": float(np.sqrt(np.mean(arr * arr))),
                "max_abs": float(np.max(np.abs(arr))),
                "mean_abs": float(np.mean(np.abs(arr))),
            }

    ang = aligned.angular_errors
    metrics["geodesic"] = {
        "rmse": float(np.sqrt(np.mean(ang * ang))),
        "max_abs": float(np.max(np.abs(ang))),
        "mean_abs": float(np.mean(np.abs(ang))),
    }

    return metrics


def worst_offenders(aligned: AlignmentResult, dimension: int, top_k: int = 25) -> pd.DataFrame:
    df = aligned.aligned_df.copy()
    if dimension == 1:
        df["error_score"] = np.abs(df["theta_0_error"].to_numpy(dtype=float))
    else:
        err_cols = [f"pos_{i}_error" for i in range(dimension + 1)]
        errs = df[err_cols].to_numpy(dtype=float)
        df["error_score"] = np.linalg.norm(errs, axis=1)
    return df.sort_values("error_score", ascending=False).head(top_k)


def compare_coordinate_files(baseline_csv: Path,
                             optimized_csv: Path,
                             dimension: int,
                             rmse_tol: float,
                             max_abs_tol: float,
                             pair_samples: int,
                             seed: int,
                             truth_csv: Optional[Path] = None,
                             debug_dir: Optional[Path] = None) -> Dict[str, object]:
    baseline = load_coordinates(baseline_csv)
    optimized = load_coordinates(optimized_csv)

    if dimension == 1:
        aligned = align_dimension_one(baseline, optimized)
    else:
        aligned = align_dimension_gt_one(baseline, optimized, dimension)

    field_metrics = compute_field_metrics(aligned, dimension)
    corr = pairwise_distance_correlation(aligned, dimension, pair_samples, seed)

    max_rmse = max(v["rmse"] for k, v in field_metrics.items() if k != "geodesic")
    max_abs = max(v["max_abs"] for k, v in field_metrics.items() if k != "geodesic")
    passed = (max_rmse <= rmse_tol) and (max_abs <= max_abs_tol)

    truth_metrics = None
    if truth_csv is not None and truth_csv.exists():
        truth = load_coordinates(truth_csv)
        if dimension == 1:
            aligned_truth = align_dimension_one(truth, optimized)
        else:
            aligned_truth = align_dimension_gt_one(truth, optimized, dimension)
        truth_metrics = compute_field_metrics(aligned_truth, dimension)

    result: Dict[str, object] = {
        "passed": bool(passed),
        "dimension": int(dimension),
        "nodes_compared": int(len(aligned.aligned_df)),
        "transform": aligned.transform,
        "field_metrics": field_metrics,
        "distance_correlation": corr,
        "tolerances": {
            "rmse_tol": float(rmse_tol),
            "max_abs_tol": float(max_abs_tol),
        },
    }
    if truth_metrics is not None:
        result["truth_alignment_metrics"] = truth_metrics

    if not passed and debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        offenders = worst_offenders(aligned, dimension)
        offenders_path = debug_dir / "worst_offenders.csv"
        offenders.to_csv(offenders_path, index=False)
        result["debug_worst_offenders"] = str(offenders_path)

    return result


def print_result_summary(result: Dict[str, object]) -> None:
    print("Coordinate comparison summary")
    print(f"  passed: {result['passed']}")
    print(f"  nodes_compared: {result['nodes_compared']}")
    print(f"  transform: {result['transform']}")
    fields: Dict[str, Dict[str, float]] = result["field_metrics"]  # type: ignore[assignment]
    for name, stats in fields.items():
        print(
            f"  {name:10s} rmse={stats['rmse']:.12g} "
            f"max_abs={stats['max_abs']:.12g} mean_abs={stats['mean_abs']:.12g}"
        )
    corr = result["distance_correlation"]
    print(
        "  pair_distance_corr "
        f"pearson={corr['pearson']:.12g} spearman={corr['spearman']:.12g} "
        f"samples={corr['pair_samples']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and optimized embedding coordinates with symmetry alignment.")
    parser.add_argument("--baseline", required=True, help="Path to baseline coordinates CSV")
    parser.add_argument("--optimized", required=True, help="Path to optimized coordinates CSV")
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--truth", default=None, help="Optional truth coordinates CSV")
    parser.add_argument("--rmse-tol", type=float, default=1e-8)
    parser.add_argument("--max-abs-tol", type=float, default=1e-6)
    parser.add_argument("--pair-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    baseline_csv = Path(args.baseline)
    optimized_csv = Path(args.optimized)
    truth_csv = Path(args.truth) if args.truth else None
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    result = compare_coordinate_files(
        baseline_csv=baseline_csv,
        optimized_csv=optimized_csv,
        dimension=args.dimension,
        rmse_tol=args.rmse_tol,
        max_abs_tol=args.max_abs_tol,
        pair_samples=args.pair_samples,
        seed=args.seed,
        truth_csv=truth_csv,
        debug_dir=debug_dir,
    )

    print_result_summary(result)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"JSON: {out_path}")

    if not result["passed"]:
        print("Comparison FAILED: tolerances exceeded.")
        if "debug_worst_offenders" in result:
            print(f"Worst offenders: {result['debug_worst_offenders']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
