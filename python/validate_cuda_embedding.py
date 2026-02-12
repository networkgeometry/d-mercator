#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PI = math.pi


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )


def sample_kappas(n: int, gamma: float, target_avg_degree: float, rng: np.random.Generator) -> np.ndarray:
    if gamma <= 2.0:
        raise ValueError("gamma must be > 2.0")
    kappa_min = target_avg_degree * (gamma - 2.0) / (gamma - 1.0)
    return kappa_min * (1.0 + rng.pareto(gamma - 1.0, size=n))


def random_unit_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(n, dim + 1))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return raw / norms


def write_hidden_variables(path: Path, kappas: np.ndarray, dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    with path.open("w", encoding="utf-8") as f:
        if dim == 1:
            theta = rng.uniform(0.0, 2.0 * PI, size=len(kappas))
            for k, t in zip(kappas, theta):
                f.write(f"{k:.17g} {t:.17g}\n")
            return
        vectors = random_unit_vectors(len(kappas), dim, rng)
        for i, k in enumerate(kappas):
            coords = " ".join(f"{x:.17g}" for x in vectors[i])
            f.write(f"{k:.17g} {coords}\n")


def load_coords(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"node_id": str})
    return df.sort_values("node_id").reset_index(drop=True)


def align_by_node_id(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame, truth_df: pd.DataFrame | None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    common = set(cpu_df["node_id"]) & set(gpu_df["node_id"])
    if truth_df is not None:
        common &= set(truth_df["node_id"])
    if not common:
        raise RuntimeError("No common nodes across CPU/GPU/truth outputs.")
    ordered = sorted(common)

    def keep(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["node_id"].isin(ordered)].sort_values("node_id").reset_index(drop=True)

    cpu = keep(cpu_df)
    gpu = keep(gpu_df)
    if truth_df is None:
        return cpu, gpu, None
    return cpu, gpu, keep(truth_df)


def parse_inf_metadata(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    patterns = {
        "beta": re.compile(r"- beta:\s+([0-9eE+.\-]+)"),
        "mu": re.compile(r"- mu:\s+([0-9eE+.\-]+)"),
        "radius_s1": re.compile(r"- radius_S1:\s+([0-9eE+.\-]+)"),
        "radius_sd": re.compile(r"- radius_S\^D:\s+([0-9eE+.\-]+)"),
    }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    out[key] = float(match.group(1))
    return out


def parse_convergence_iterations(path: Path) -> List[int]:
    pattern = re.compile(r"Convergence reached after\s+(\d+)\s+iterations\.")
    out: List[int] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                out.append(int(match.group(1)))
    return out


def read_edge_set(path: Path, node_to_idx: Dict[str, int]) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            n1, n2 = parts[0], parts[1]
            if n1 not in node_to_idx or n2 not in node_to_idx:
                continue
            i, j = node_to_idx[n1], node_to_idx[n2]
            if i == j:
                continue
            if i > j:
                i, j = j, i
            edges.add((i, j))
    return edges


def circular_signed_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def align_angles(reference: np.ndarray, target: np.ndarray, n_shifts: int = 4096) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    shifts = np.linspace(0.0, 2.0 * PI, num=n_shifts, endpoint=False)

    def evaluate(candidate: np.ndarray):
        best_mse = float("inf")
        best_shift = 0.0
        best_aligned = candidate
        best_error = circular_signed_diff(candidate, reference)
        for shift in shifts:
            aligned = np.mod(candidate + shift, 2.0 * PI)
            error = circular_signed_diff(aligned, reference)
            mse = float(np.mean(error * error))
            if mse < best_mse:
                best_mse = mse
                best_shift = float(shift)
                best_aligned = aligned
                best_error = error
        return best_aligned, best_error, best_mse, best_shift

    direct_aligned, direct_err, direct_mse, direct_shift = evaluate(np.mod(target, 2.0 * PI))
    reflected_aligned, reflected_err, reflected_mse, reflected_shift = evaluate(np.mod(-target, 2.0 * PI))

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


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def procrustes_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    reference = normalize_rows(reference)
    target = normalize_rows(target)
    m = target.T @ reference
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return normalize_rows(target @ r)


def compute_angle_d_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    norm1 = float(np.dot(v1, v1))
    norm2 = float(np.dot(v2, v2))
    norm1 /= math.sqrt(norm1)
    norm2 /= math.sqrt(norm2)
    ratio = dot / (norm1 * norm2)
    if abs(ratio - 1.0) < 1e-10:
        return 0.0
    return float(math.acos(ratio))


def compute_objective_s1(theta: np.ndarray,
                         kappa: np.ndarray,
                         beta: float,
                         mu: float,
                         edges: Set[Tuple[int, int]]) -> float:
    n = len(theta)
    prefactor = n / (2.0 * PI * mu)
    objective = 0.0
    for i in range(n):
        t1 = theta[i]
        k1 = kappa[i]
        for j in range(i + 1, n):
            da = PI - abs(PI - abs(t1 - theta[j]))
            fraction = (prefactor * da) / (k1 * kappa[j])
            if (i, j) in edges:
                objective += -beta * math.log(fraction)
            else:
                objective += -math.log(1.0 + math.pow(fraction, -beta))
    return objective


def compute_objective_sd(positions: np.ndarray,
                         kappa: np.ndarray,
                         dim: int,
                         beta: float,
                         mu: float,
                         radius: float,
                         edges: Set[Tuple[int, int]]) -> float:
    n = positions.shape[0]
    inv_dim = 1.0 / float(dim)
    objective = 0.0
    for i in range(n):
        pos1 = positions[i]
        k1 = kappa[i]
        for j in range(i + 1, n):
            dtheta = compute_angle_d_vectors(pos1, positions[j])
            chi = radius * dtheta / math.pow(mu * k1 * kappa[j], inv_dim)
            prob = 1.0 / (1.0 + math.pow(chi, beta))
            if (i, j) in edges:
                objective += math.log(prob)
            else:
                objective += math.log(1.0 - prob)
    return objective


def save_dim1_plots(cpu_theta: np.ndarray,
                    gpu_theta_aligned: np.ndarray,
                    errors: np.ndarray,
                    truth_theta: np.ndarray | None,
                    plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(cpu_theta, gpu_theta_aligned, s=10, alpha=0.6)
    plt.plot([0, 2 * PI], [0, 2 * PI], "k--", linewidth=1)
    plt.xlim(0, 2 * PI)
    plt.ylim(0, 2 * PI)
    plt.xlabel("CPU inferred angle")
    plt.ylabel("GPU inferred angle (aligned)")
    plt.title("CPU vs GPU Angles")
    plt.tight_layout()
    plt.savefig(plots_dir / "cpu_vs_gpu_angles_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(np.abs(errors), bins=40, edgecolor="black")
    plt.xlabel("Absolute angular error (rad)")
    plt.ylabel("Count")
    plt.title("CPU vs GPU Angular Error Histogram")
    plt.tight_layout()
    plt.savefig(plots_dir / "cpu_vs_gpu_angle_error_hist.png", dpi=160)
    plt.close()

    if truth_theta is not None:
        aligned_cpu, _, _ = align_angles(truth_theta, cpu_theta)
        aligned_gpu, _, _ = align_angles(truth_theta, gpu_theta_aligned)
        plt.figure(figsize=(6, 6))
        plt.scatter(truth_theta, aligned_cpu, s=10, alpha=0.5, label="CPU")
        plt.scatter(truth_theta, aligned_gpu, s=10, alpha=0.5, label="GPU")
        plt.plot([0, 2 * PI], [0, 2 * PI], "k--", linewidth=1)
        plt.xlim(0, 2 * PI)
        plt.ylim(0, 2 * PI)
        plt.xlabel("Ground-truth angle")
        plt.ylabel("Inferred angle (aligned)")
        plt.title("Ground Truth vs Inferred Angles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "truth_vs_inferred_angles_scatter.png", dpi=160)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CPU vs CUDA embedding parity on synthetic SD networks.")
    parser.add_argument("--build-dir", default="build", help="Build directory containing `generate_sd` and `embed_sd`.")
    parser.add_argument("--output-dir", default="validation_output/cuda_parity", help="Directory for outputs.")
    parser.add_argument("--dimension", type=int, default=1, help="Embedding dimension.")
    parser.add_argument("--nodes", type=int, default=220, help="Number of synthetic nodes.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--gamma", type=float, default=2.7, help="Power-law exponent for kappa sampling.")
    parser.add_argument("--target-avg-degree", type=float, default=10.0, help="Target average hidden degree.")
    parser.add_argument("--beta", type=float, default=None, help="Custom beta. Default is 2*dimension.")
    parser.add_argument("--objective-rel-tol", type=float, default=1e-8, help="Acceptance threshold for objective relative error.")
    parser.add_argument("--max-angle-dev-tol", type=float, default=1e-6, help="Acceptance threshold for max coordinate-angle deviation.")
    args = parser.parse_args()

    if args.dimension < 1:
        raise ValueError("dimension must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (repo_root / build_dir).resolve()

    generate_sd = build_dir / "generate_sd"
    embed_sd = build_dir / "embed_sd"
    if not generate_sd.exists() or not embed_sd.exists():
        raise RuntimeError(
            f"Missing executables. Expected `{generate_sd}` and `{embed_sd}`."
        )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    beta = float(args.beta if args.beta is not None else (2.0 * args.dimension))
    rng = np.random.default_rng(args.seed)
    kappas = sample_kappas(args.nodes, args.gamma, args.target_avg_degree, rng)

    hidden_path = output_dir / "synthetic.hidden.txt"
    write_hidden_variables(hidden_path, kappas, args.dimension, args.seed + 11)

    gen_root = output_dir / "synthetic_sd"
    cpu_root = output_dir / "cpu_embedding"
    gpu_root = output_dir / "gpu_embedding"
    edge_path = Path(str(gen_root) + ".edge")
    truth_csv_path = Path(str(gen_root) + ".truth.csv")
    cpu_csv_path = Path(str(cpu_root) + ".coords.csv")
    gpu_csv_path = Path(str(gpu_root) + ".coords.csv")
    cpu_inf_coord = Path(str(cpu_root) + ".inf_coord")
    gpu_inf_coord = Path(str(gpu_root) + ".inf_coord")
    cpu_log = Path(str(cpu_root) + ".inf_log")
    gpu_log = Path(str(gpu_root) + ".inf_log")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"

    run_command([
        str(generate_sd),
        "-d", str(args.dimension),
        "-b", f"{beta:.17g}",
        "-s", str(args.seed),
        "-t",
        "-o", str(gen_root),
        str(hidden_path),
    ], cwd=repo_root, env=env)

    run_command([
        str(embed_sd),
        "-d", str(args.dimension),
        "-b", f"{beta:.17g}",
        "-s", str(args.seed),
        "-M", "optimized",
        "-o", str(cpu_root),
        str(edge_path),
    ], cwd=repo_root, env=env)

    run_command([
        str(embed_sd),
        "-d", str(args.dimension),
        "-b", f"{beta:.17g}",
        "-s", str(args.seed),
        "-M", "optimized",
        "-G",
        "-D",
        "-o", str(gpu_root),
        str(edge_path),
    ], cwd=repo_root, env=env)

    cpu_df = load_coords(cpu_csv_path)
    gpu_df = load_coords(gpu_csv_path)
    truth_df = load_coords(truth_csv_path) if truth_csv_path.exists() else None
    cpu_df, gpu_df, truth_df = align_by_node_id(cpu_df, gpu_df, truth_df)

    node_ids = cpu_df["node_id"].tolist()
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_set = read_edge_set(edge_path, node_to_idx)

    cpu_meta = parse_inf_metadata(cpu_inf_coord)
    gpu_meta = parse_inf_metadata(gpu_inf_coord)
    beta_cpu = cpu_meta.get("beta", beta)
    mu_cpu = cpu_meta.get("mu", 1.0)
    mu_gpu = gpu_meta.get("mu", mu_cpu)
    radius_sd = cpu_meta.get("radius_sd", 0.0)

    metrics: Dict[str, float | int | str] = {
        "dimension": args.dimension,
        "nodes": len(cpu_df),
        "seed": args.seed,
        "beta_cpu": beta_cpu,
        "mu_cpu": mu_cpu,
        "mu_gpu": mu_gpu,
    }

    cpu_kappa = cpu_df["kappa"].to_numpy(dtype=float)
    gpu_kappa = gpu_df["kappa"].to_numpy(dtype=float)
    metrics["kappa_max_abs_diff"] = float(np.max(np.abs(cpu_kappa - gpu_kappa)))

    if args.dimension == 1:
        cpu_theta = np.mod(cpu_df["theta_0"].to_numpy(dtype=float), 2.0 * PI)
        gpu_theta = np.mod(gpu_df["theta_0"].to_numpy(dtype=float), 2.0 * PI)
        gpu_theta_aligned, angle_errors, alignment = align_angles(cpu_theta, gpu_theta)
        metrics["alignment_reflection"] = alignment["reflection"]
        metrics["alignment_shift"] = alignment["shift"]
        metrics["angle_rmse"] = float(np.sqrt(np.mean(angle_errors * angle_errors)))
        metrics["angle_max_abs"] = float(np.max(np.abs(angle_errors)))

        objective_cpu = compute_objective_s1(cpu_theta, cpu_kappa, beta_cpu, mu_cpu, edge_set)
        objective_gpu = compute_objective_s1(gpu_theta_aligned, gpu_kappa, beta_cpu, mu_cpu, edge_set)
        save_dim1_plots(
            cpu_theta,
            gpu_theta_aligned,
            angle_errors,
            truth_df["theta_0"].to_numpy(dtype=float) if truth_df is not None and "theta_0" in truth_df.columns else None,
            plots_dir,
        )
    else:
        coord_cols = [f"pos_{i}" for i in range(args.dimension + 1)]
        cpu_pos = cpu_df[coord_cols].to_numpy(dtype=float)
        gpu_pos = gpu_df[coord_cols].to_numpy(dtype=float)
        cpu_pos_n = normalize_rows(cpu_pos)
        gpu_pos_n = normalize_rows(gpu_pos)
        gpu_pos_aligned = procrustes_align(cpu_pos_n, gpu_pos_n)
        dots = np.sum(cpu_pos_n * gpu_pos_aligned, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angle_errors = np.arccos(dots)
        metrics["angle_rmse"] = float(np.sqrt(np.mean(angle_errors * angle_errors)))
        metrics["angle_max_abs"] = float(np.max(np.abs(angle_errors)))

        objective_cpu = compute_objective_sd(cpu_pos, cpu_kappa, args.dimension, beta_cpu, mu_cpu, radius_sd, edge_set)
        objective_gpu = compute_objective_sd(gpu_pos_aligned, gpu_kappa, args.dimension, beta_cpu, mu_cpu, radius_sd, edge_set)

        plt.figure(figsize=(6, 6))
        plt.scatter(cpu_pos_n[:, 0], gpu_pos_aligned[:, 0], s=10, alpha=0.6)
        plt.xlabel("CPU normalized pos_0")
        plt.ylabel("GPU normalized pos_0 (aligned)")
        plt.title("CPU vs GPU Position Correlation")
        plt.tight_layout()
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / "cpu_vs_gpu_pos0_scatter.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(np.abs(angle_errors), bins=40, edgecolor="black")
        plt.xlabel("Absolute angular error (rad)")
        plt.ylabel("Count")
        plt.title("CPU vs GPU Vector-Angle Error Histogram")
        plt.tight_layout()
        plt.savefig(plots_dir / "cpu_vs_gpu_vector_angle_error_hist.png", dpi=160)
        plt.close()

    objective_rel_err = abs(objective_gpu - objective_cpu) / max(1.0, abs(objective_cpu))
    metrics["objective_cpu"] = float(objective_cpu)
    metrics["objective_gpu"] = float(objective_gpu)
    metrics["objective_rel_err"] = float(objective_rel_err)

    cpu_conv = parse_convergence_iterations(cpu_log)
    gpu_conv = parse_convergence_iterations(gpu_log)
    metrics["cpu_convergence_iterations"] = cpu_conv
    metrics["gpu_convergence_iterations"] = gpu_conv
    metrics["convergence_sequence_match"] = int(cpu_conv == gpu_conv)

    metrics_path = output_dir / "cuda_validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plots saved to: {plots_dir}")

    failures = []
    if objective_rel_err > args.objective_rel_tol:
        failures.append(
            f"objective_rel_err={objective_rel_err:.3e} exceeds tolerance {args.objective_rel_tol:.3e}"
        )
    if float(metrics["angle_max_abs"]) > args.max_angle_dev_tol:
        failures.append(
            f"angle_max_abs={float(metrics['angle_max_abs']):.3e} exceeds tolerance {args.max_angle_dev_tol:.3e}"
        )
    if failures:
        print("Validation FAILED:", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print("Validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
