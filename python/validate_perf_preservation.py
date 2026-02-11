#!/usr/bin/env python3

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Install numpy, pandas, matplotlib.")


def run(cmd, cwd: Path) -> None:
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def write_hidden_variables(path: Path, nodes: int, dimension: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    kappas = rng.uniform(2.0, 12.0, size=nodes)

    with path.open("w", encoding="utf-8") as f:
        for i in range(nodes):
            if dimension == 1:
                theta = rng.uniform(0.0, 2.0 * math.pi)
                f.write(f"{kappas[i]:.17g} {theta:.17g}\n")
            else:
                vec = rng.normal(size=dimension + 1)
                norm = float(np.linalg.norm(vec))
                if norm == 0.0:
                    vec[0] = 1.0
                    norm = 1.0
                vec = vec / norm
                coords = " ".join(f"{x:.17g}" for x in vec)
                f.write(f"{kappas[i]:.17g} {coords}\n")


def circular_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def compare_outputs(baseline_csv: Path, optimized_csv: Path, dimension: int) -> dict:
    baseline_df = pd.read_csv(baseline_csv)
    optimized_df = pd.read_csv(optimized_csv)

    merged = baseline_df.merge(optimized_df, on="node_id", suffixes=("_base", "_opt"), how="inner")
    if merged.empty:
        raise RuntimeError("No common node_id rows between baseline and optimized outputs")

    if dimension == 1:
        fields = ["kappa", "r", "theta_0"]
        coordinate_fields = ["theta_0"]
    else:
        fields = ["kappa", "r"] + [f"pos_{i}" for i in range(dimension + 1)]
        coordinate_fields = [f"pos_{i}" for i in range(dimension + 1)]

    field_metrics = {}
    coord_deltas = []

    for field in fields:
        base = merged[f"{field}_base"].to_numpy(dtype=float)
        opt = merged[f"{field}_opt"].to_numpy(dtype=float)
        if field == "theta_0":
            delta = circular_delta(opt, base)
        else:
            delta = opt - base

        field_metrics[field] = {
            "rmse": rmse(delta),
            "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))),
        }

        if field in coordinate_fields:
            coord_deltas.append(delta)

    flattened_coords = np.concatenate(coord_deltas)
    coordinate_rmse = rmse(flattened_coords)

    return {
        "nodes_compared": int(len(merged)),
        "dimension": int(dimension),
        "coordinate_rmse": coordinate_rmse,
        "fields": field_metrics,
    }


def plot_scatter(baseline_csv: Path, optimized_csv: Path, dimension: int, out_png: Path) -> None:
    baseline_df = pd.read_csv(baseline_csv)
    optimized_df = pd.read_csv(optimized_csv)
    merged = baseline_df.merge(optimized_df, on="node_id", suffixes=("_base", "_opt"), how="inner")
    if merged.empty:
        raise RuntimeError("No common node_id rows to plot")

    field = "theta_0" if dimension == 1 else "pos_0"
    x = merged[f"{field}_base"].to_numpy(dtype=float)
    y = merged[f"{field}_opt"].to_numpy(dtype=float)

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=12, alpha=0.65)
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel(f"baseline {field}")
    plt.ylabel(f"optimized {field}")
    plt.title(f"Baseline vs Optimized ({field})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate perf-preserving refactor (baseline vs optimized mode).")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--output-dir", default="validation_output/perf_preservation")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--beta", type=float, default=4.5)
    parser.add_argument("--rmse-tol", type=float, default=1e-7)
    parser.add_argument("--max-abs-tol", type=float, default=1e-6)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = (repo_root / args.build_dir).resolve()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dimension < 1:
        raise SystemExit("--dimension must be >= 1")

    if not args.skip_build:
        run(["cmake", "-S", str(repo_root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], repo_root)
        run(["cmake", "--build", str(build_dir), "-j"], repo_root)

    generate_bin = build_dir / "generate_sd"
    embed_bin = build_dir / "embed_sd"
    if not generate_bin.exists():
        raise SystemExit(f"Missing binary: {generate_bin}")
    if not embed_bin.exists():
        raise SystemExit(f"Missing binary: {embed_bin}")

    hidden_path = out_dir / "hidden_variables.txt"
    write_hidden_variables(hidden_path, args.nodes, args.dimension, args.seed)

    sd_root = out_dir / "synthetic_sd"
    run([
        str(generate_bin),
        "-d", str(args.dimension),
        "-b", str(args.beta),
        "-s", str(args.seed),
        "-o", str(sd_root),
        "-t",
        str(hidden_path),
    ], repo_root)

    edge_file = Path(str(sd_root) + ".edge")
    if not edge_file.exists():
        raise SystemExit(f"Missing generated edge list: {edge_file}")

    baseline_root = out_dir / "embed_baseline"
    optimized_root = out_dir / "embed_optimized"

    run([
        str(embed_bin),
        "-q",
        "-d", str(args.dimension),
        "-s", str(args.seed),
        "-b", str(args.beta),
        "-M", "baseline",
        "-o", str(baseline_root),
        str(edge_file),
    ], repo_root)

    run([
        str(embed_bin),
        "-q",
        "-d", str(args.dimension),
        "-s", str(args.seed),
        "-b", str(args.beta),
        "-M", "optimized",
        "-o", str(optimized_root),
        str(edge_file),
    ], repo_root)

    baseline_csv = Path(str(baseline_root) + ".coords.csv")
    optimized_csv = Path(str(optimized_root) + ".coords.csv")
    if not baseline_csv.exists() or not optimized_csv.exists():
        raise SystemExit("Expected .coords.csv outputs were not produced")

    metrics = compare_outputs(baseline_csv, optimized_csv, args.dimension)

    scatter_path = out_dir / "baseline_vs_optimized_scatter.png"
    plot_scatter(baseline_csv, optimized_csv, args.dimension, scatter_path)

    metrics_path = out_dir / "perf_preservation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nPerformance preservation metrics")
    print(f"  Nodes compared: {metrics['nodes_compared']}")
    print(f"  Coordinate RMSE: {metrics['coordinate_rmse']:.12g}")

    max_field_abs = 0.0
    for field, vals in metrics["fields"].items():
        max_field_abs = max(max_field_abs, vals["max_abs"])
        print(
            f"  {field:8s} rmse={vals['rmse']:.12g} "
            f"max_abs={vals['max_abs']:.12g} mean_abs={vals['mean_abs']:.12g}"
        )

    passed = (metrics["coordinate_rmse"] <= args.rmse_tol) and (max_field_abs <= args.max_abs_tol)
    print(f"\nTolerances: rmse_tol={args.rmse_tol} max_abs_tol={args.max_abs_tol}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Scatter plot: {scatter_path}")

    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
