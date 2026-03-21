#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


TIMING_KEYS = [
    "total_time_ms",
    "initialization_ms",
    "parameter_inference_ms",
    "initial_positions_ms",
    "refining_positions_ms",
    "adjusting_kappas_ms",
    "io_ms",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark exact vs negative-sampled refinement across CPU/GPU backends."
    )
    parser.add_argument("--cpu-build-dir", help="Build directory configured with -DUSE_CUDA=OFF.")
    parser.add_argument("--gpu-build-dir", help="Build directory configured with -DUSE_CUDA=ON.")
    parser.add_argument(
        "--generator-build-dir",
        default=None,
        help="Build directory containing generate_sd. Defaults to cpu build dir when available, otherwise gpu build dir.",
    )
    parser.add_argument("--out-dir", default="benchmark_runs/negative_sampling_cpu_gpu")
    parser.add_argument("--dimensions", default="1,2")
    parser.add_argument("--sizes", default="1000,2000,5000,10000")
    parser.add_argument("--sample-counts", default="16,64,256")
    parser.add_argument("--backends", default="cpu,gpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--beta-mode",
        choices=("fixed", "two-times-dim"),
        default="two-times-dim",
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--mean-degree", type=float, default=14.0)
    parser.add_argument("--quality-size-limit", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--omp-threads", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_int_list(raw: str) -> List[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_backends(raw: str) -> List[str]:
    backends: List[str] = []
    for token in raw.split(","):
        backend = token.strip().lower()
        if backend not in ("cpu", "gpu"):
            continue
        if backend not in backends:
            backends.append(backend)
    if not backends:
        raise SystemExit("No valid backends provided. Allowed values: cpu,gpu")
    return backends


def resolve_beta(dim: int, args) -> float:
    if args.beta_mode == "two-times-dim":
        return 2.0 * dim
    return args.beta


def required_binary(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing executable: {path}")
    return path


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str], capture_output: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    kwargs: Dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
        "check": True,
        "text": True,
    }
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(cmd, **kwargs)


def parse_timing_json(stdout: str, stderr: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines() + stderr.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "total_time_ms" in payload:
            return payload
    return {}


def write_hidden_variables(path: Path, nodes: int, dimension: int, gamma: float, mean_degree: float, seed: int) -> None:
    rng = random.Random(seed)
    kappa_0 = (
        (1.0 - 1.0 / nodes)
        / (1.0 - nodes ** ((2.0 - gamma) / (gamma - 1.0)))
        * (gamma - 2.0)
        / (gamma - 1.0)
        * mean_degree
    )
    kappa_c = kappa_0 * nodes ** (1.0 / (gamma - 1.0))
    scale = 1.0 - (kappa_c / kappa_0) ** (1.0 - gamma)

    with path.open("w", encoding="utf-8") as handle:
        for _ in range(nodes):
            u = rng.random()
            kappa = kappa_0 * (1.0 - u * scale) ** (1.0 / (1.0 - gamma))
            if dimension == 1:
                theta = 2.0 * math.pi * rng.random()
                handle.write(f"{kappa:.17g} {theta:.17g}\n")
                continue

            coords = [rng.gauss(0.0, 1.0) for _ in range(dimension + 1)]
            norm = math.sqrt(sum(value * value for value in coords))
            coords = [value / norm for value in coords]
            handle.write(f"{kappa:.17g} {' '.join(f'{value:.17g}' for value in coords)}\n")


def extract_gcc_edge_list(edge_path: Path, out_path: Path) -> Path:
    graph = nx.read_edgelist(edge_path, comments="#", data=False)
    gcc_nodes = max(nx.connected_components(graph), key=len)
    gcc_graph = graph.subgraph(gcc_nodes).copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# source target\n")
        for source, target in gcc_graph.edges():
            handle.write(f"{source} {target}\n")
    return out_path


def parse_inferred_coordinates(path: Path, dim: int) -> Dict[str, Any]:
    beta = None
    mu = None
    names: List[str] = []
    kappas: List[float] = []
    theta: List[float] = []
    positions: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if stripped.startswith("#   - beta:"):
                    beta = float(stripped.split()[-1])
                elif stripped.startswith("#   - mu:"):
                    mu = float(stripped.split()[-1])
                continue
            fields = stripped.split()
            names.append(fields[0])
            kappas.append(float(fields[1]))
            if dim == 1:
                theta.append(float(fields[2]))
            else:
                positions.append([float(value) for value in fields[3 : 3 + dim + 1]])
    if beta is None or mu is None:
        raise RuntimeError(f"Could not parse beta/mu from {path}")
    return {
        "beta": beta,
        "mu": mu,
        "names": names,
        "kappas": np.array(kappas, dtype=float),
        "theta": np.array(theta, dtype=float) if dim == 1 else None,
        "positions": np.array(positions, dtype=float) if dim > 1 else None,
    }


def parse_generated_coordinates(path: Path, dim: int) -> Dict[str, Any]:
    names: List[str] = []
    kappas: List[float] = []
    theta: List[float] = []
    positions: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            names.append(fields[0])
            kappas.append(float(fields[1]))
            if dim == 1:
                theta.append(float(fields[2]))
            else:
                positions.append([float(value) for value in fields[3 : 3 + dim + 1]])
    return {
        "names": names,
        "kappas": np.array(kappas, dtype=float),
        "theta": np.array(theta, dtype=float) if dim == 1 else None,
        "positions": np.array(positions, dtype=float) if dim > 1 else None,
    }


def reorder_generated_to_inferred(generated: Dict[str, Any], inferred: Dict[str, Any]) -> Dict[str, Any]:
    generated_index = {name: index for index, name in enumerate(generated["names"])}
    order = np.array([generated_index[name] for name in inferred["names"]], dtype=int)
    return {
        "names": inferred["names"],
        "kappas": generated["kappas"][order],
        "theta": None if generated["theta"] is None else generated["theta"][order],
        "positions": None if generated["positions"] is None else generated["positions"][order],
    }


def load_edge_list(edge_path: Path, vertex_to_index: Dict[str, int]) -> np.ndarray:
    edges = []
    with edge_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            source, target = stripped.split()[:2]
            if source == target:
                continue
            v1 = vertex_to_index[source]
            v2 = vertex_to_index[target]
            edges.append((min(v1, v2), max(v1, v2)))
    return np.array(edges, dtype=np.int64)


def compute_radius(dim: int, size: int) -> float:
    inside = size / (2 * math.pi ** ((dim + 1) / 2.0)) * math.gamma((dim + 1) / 2.0)
    return inside ** (1.0 / dim)


def upper_triangle_sum(matrix: np.ndarray) -> float:
    return float(np.triu(matrix, k=1).sum())


def compute_exact_objective_1d(kappa: np.ndarray,
                               theta: np.ndarray,
                               mu: float,
                               beta: float,
                               edges: np.ndarray,
                               block_size: int) -> float:
    size = len(kappa)
    total = 0.0
    for i_start in range(0, size, block_size):
        i_stop = min(size, i_start + block_size)
        theta_i = theta[i_start:i_stop]
        kappa_i = kappa[i_start:i_stop]
        for j_start in range(i_start, size, block_size):
            j_stop = min(size, j_start + block_size)
            theta_j = theta[j_start:j_stop]
            kappa_j = kappa[j_start:j_stop]
            theta_diff = np.abs(theta_i[:, None] - theta_j[None, :])
            angular_distance = math.pi - np.abs(math.pi - theta_diff)
            scale = 2 * math.pi * mu * np.outer(kappa_i, kappa_j)
            fraction = np.divide(size * angular_distance, scale, out=np.ones_like(scale), where=scale > 0)
            fraction = np.maximum(fraction, 1e-300)
            log_fraction = np.log(fraction)
            non_neighbor = -np.logaddexp(0.0, -beta * log_fraction)
            if i_start == j_start:
                total += upper_triangle_sum(non_neighbor)
            else:
                total += float(non_neighbor.sum())

    if edges.size > 0:
        v1 = edges[:, 0]
        v2 = edges[:, 1]
        theta_diff = np.abs(theta[v1] - theta[v2])
        angular_distance = math.pi - np.abs(math.pi - theta_diff)
        scale = 2 * math.pi * mu * kappa[v1] * kappa[v2]
        fraction = np.maximum(size * angular_distance / scale, 1e-300)
        total += float((-beta * np.log(fraction)).sum())
    return total


def compute_exact_objective_dim(dim: int,
                                positions: np.ndarray,
                                kappa: np.ndarray,
                                mu: float,
                                beta: float,
                                edges: np.ndarray,
                                block_size: int) -> float:
    size = len(kappa)
    radius = compute_radius(dim, size)
    normalized = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    total = 0.0

    for i_start in range(0, size, block_size):
        i_stop = min(size, i_start + block_size)
        pos_i = normalized[i_start:i_stop]
        kappa_i = kappa[i_start:i_stop]
        for j_start in range(i_start, size, block_size):
            j_stop = min(size, j_start + block_size)
            pos_j = normalized[j_start:j_stop]
            kappa_j = kappa[j_start:j_stop]
            cosine = np.clip(pos_i @ pos_j.T, -1.0, 1.0)
            angular_distance = np.arccos(cosine)
            chi_scale = np.power(np.maximum(mu * np.outer(kappa_i, kappa_j), 1e-300), 1.0 / dim)
            chi = radius * angular_distance / chi_scale
            probability = 1.0 / (1.0 + np.power(chi, beta))
            probability = np.clip(probability, 1e-300, 1.0 - 1e-15)
            non_neighbor = np.log(1.0 - probability)
            if i_start == j_start:
                total += upper_triangle_sum(non_neighbor)
            else:
                total += float(non_neighbor.sum())

    if edges.size > 0:
        v1 = edges[:, 0]
        v2 = edges[:, 1]
        cosine = np.clip(np.sum(normalized[v1] * normalized[v2], axis=1), -1.0, 1.0)
        angular_distance = np.arccos(cosine)
        chi_scale = np.power(np.maximum(mu * kappa[v1] * kappa[v2], 1e-300), 1.0 / dim)
        chi = radius * angular_distance / chi_scale
        probability = 1.0 / (1.0 + np.power(chi, beta))
        probability = np.clip(probability, 1e-300, 1.0 - 1e-15)
        total += float(np.log(probability).sum())
    return total


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2 * math.pi) - math.pi


def compute_c_score(theta_original: np.ndarray, theta_inferred: np.ndarray, block_size: int) -> float:
    size = len(theta_original)
    total_pairs = size * (size - 1) // 2
    matches = 0
    for i_start in range(0, size, block_size):
        i_stop = min(size, i_start + block_size)
        orig_i = theta_original[i_start:i_stop]
        inf_i = theta_inferred[i_start:i_stop]
        for j_start in range(i_start, size, block_size):
            j_stop = min(size, j_start + block_size)
            orig_j = theta_original[j_start:j_stop]
            inf_j = theta_inferred[j_start:j_stop]
            diff_orig = wrap_to_pi(orig_j[None, :] - orig_i[:, None])
            diff_inf = wrap_to_pi(inf_j[None, :] - inf_i[:, None])
            same_direction = np.sign(diff_orig) == np.sign(diff_inf)
            if i_start == j_start:
                matches += int(np.triu(same_direction, k=1).sum())
            else:
                matches += int(same_direction.sum())
    return max(matches, total_pairs - matches) / total_pairs


def compute_aligned_coordinate_correlation(original_positions: np.ndarray, inferred_positions: np.ndarray) -> float:
    original = original_positions / np.linalg.norm(original_positions, axis=1, keepdims=True)
    inferred = inferred_positions / np.linalg.norm(inferred_positions, axis=1, keepdims=True)
    covariance = inferred.T @ original
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    aligned = inferred @ rotation
    return float(np.corrcoef(original.reshape(-1), aligned.reshape(-1))[0, 1])


def compute_quality_metrics(edge_path: Path, gen_coord_path: Path, inf_coord_path: Path, dim: int, block_size: int) -> Dict[str, Any]:
    inferred = parse_inferred_coordinates(inf_coord_path, dim)
    generated = reorder_generated_to_inferred(parse_generated_coordinates(gen_coord_path, dim), inferred)
    vertex_to_index = {name: index for index, name in enumerate(inferred["names"])}
    edges = load_edge_list(edge_path, vertex_to_index)

    if dim == 1:
        return {
            "objective": compute_exact_objective_1d(
                inferred["kappas"], inferred["theta"], inferred["mu"], inferred["beta"], edges, block_size
            ),
            "c_score": compute_c_score(generated["theta"], inferred["theta"], block_size),
            "aligned_coord_corr": None,
        }

    return {
        "objective": compute_exact_objective_dim(
            dim, inferred["positions"], inferred["kappas"], inferred["mu"], inferred["beta"], edges, block_size
        ),
        "c_score": None,
        "aligned_coord_corr": compute_aligned_coordinate_correlation(generated["positions"], inferred["positions"]),
    }


def benchmark_embedding(backend: str,
                        embed_bin: Path,
                        edge_path: Path,
                        out_root: Path,
                        dimension: int,
                        beta: float,
                        seed: int,
                        negative_samples: int,
                        env: Dict[str, str]) -> Dict[str, Any]:
    cmd = [
        str(embed_bin),
        "-q",
        "-d", str(dimension),
        "-s", str(seed),
        "-b", str(beta),
        "-j", str(negative_samples),
        "--timing_json",
    ]
    if backend == "cpu":
        cmd.extend(["-C", "-D"])
    elif backend == "gpu":
        cmd.extend(["-G", "-D"])
    else:
        raise ValueError(f"unsupported backend {backend}")
    cmd.extend(["-o", str(out_root), str(edge_path)])

    t0 = time.perf_counter()
    proc = run_cmd(cmd, cwd=edge_path.parent, env=env, capture_output=True)
    t1 = time.perf_counter()

    timing = parse_timing_json(proc.stdout, proc.stderr)
    timing["wall_time_ms"] = (t1 - t0) * 1000.0
    return timing


def series_label(record: Dict[str, Any]) -> str:
    sample_label = "exact" if record["sample_count"] == "exact" else f"S={record['sample_count']}"
    return f"{record['backend']}/{sample_label}"


def plot_metric(records: List[Dict[str, Any]],
                dimensions: List[int],
                output_path: Path,
                metric_key: str,
                title: str,
                ylabel: str,
                include_only_with_metric: bool = False,
                y_scale: str = "linear") -> None:
    fig, axes = plt.subplots(1, len(dimensions), figsize=(7 * len(dimensions), 5), squeeze=False)
    palette = ["#111111", "#0b6e4f", "#b85c38", "#5d576b", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for axis, dim in zip(axes[0], dimensions):
        dim_records = [record for record in records if record["dimension"] == dim]
        labels = sorted({series_label(record) for record in dim_records})
        color_map = {label: palette[index % len(palette)] for index, label in enumerate(labels)}
        for label in labels:
            series = sorted(
                [record for record in dim_records if series_label(record) == label],
                key=lambda item: item["size"],
            )
            if include_only_with_metric:
                series = [record for record in series if record.get(metric_key) is not None]
            if not series:
                continue
            axis.plot(
                [record["size"] for record in series],
                [record[metric_key] for record in series],
                marker="o",
                linewidth=2,
                color=color_map[label],
                label=label,
            )
        axis.set_title(f"D={dim}")
        axis.set_xlabel("N")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.set_xscale("log")
        if y_scale != "linear":
            axis.set_yscale(y_scale)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    selected_backends = parse_backends(args.backends)
    if "cpu" in selected_backends and not args.cpu_build_dir:
        raise SystemExit("--cpu-build-dir is required when --backends includes cpu")
    if "gpu" in selected_backends and not args.gpu_build_dir:
        raise SystemExit("--gpu-build-dir is required when --backends includes gpu")

    repo_root = Path(__file__).resolve().parents[1]
    cpu_build_dir = Path(args.cpu_build_dir).resolve() if args.cpu_build_dir else None
    gpu_build_dir = Path(args.gpu_build_dir).resolve() if args.gpu_build_dir else None
    if args.generator_build_dir:
        generator_build_dir = Path(args.generator_build_dir).resolve()
    elif cpu_build_dir is not None:
        generator_build_dir = cpu_build_dir
    elif gpu_build_dir is not None:
        generator_build_dir = gpu_build_dir
    else:
        raise SystemExit("Unable to resolve generator build dir. Provide --generator-build-dir.")

    generate_bin = required_binary(generator_build_dir / "generate_sd")
    embed_bins: Dict[str, Path] = {}
    if "cpu" in selected_backends:
        embed_bins["cpu"] = required_binary(cpu_build_dir / "embed_sd")
    if "gpu" in selected_backends:
        embed_bins["gpu"] = required_binary(gpu_build_dir / "embed_sd")

    dimensions = parse_int_list(args.dimensions)
    sizes = parse_int_list(args.sizes)
    sample_counts = parse_int_list(args.sample_counts)

    out_dir = Path(args.out_dir).resolve()
    artifacts_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(args.omp_threads)

    records: List[Dict[str, Any]] = []

    for dimension in dimensions:
        beta = resolve_beta(dimension, args)
        for size in sizes:
            graph_seed = args.seed + dimension * 1_000_000 + size
            case_dir = artifacts_dir / f"d{dimension}_n{size}_seed{graph_seed}"
            case_dir.mkdir(parents=True, exist_ok=True)

            hidden_path = case_dir / "hidden_variables.txt"
            graph_root = case_dir / "synthetic_sd"
            edge_path = Path(str(graph_root) + ".edge")
            gcc_edge_path = Path(str(graph_root) + "_GC.edge")
            gen_coord_path = Path(str(graph_root) + ".gen_coord")

            if args.force or not hidden_path.exists():
                write_hidden_variables(hidden_path, size, dimension, args.gamma, args.mean_degree, graph_seed)

            if args.force or not edge_path.exists() or not gen_coord_path.exists():
                run_cmd(
                    [
                        str(generate_bin),
                        "-d", str(dimension),
                        "-b", str(beta),
                        "-s", str(graph_seed),
                        "-v",
                        "-t",
                        "-o", str(graph_root),
                        str(hidden_path),
                    ],
                    cwd=repo_root,
                    env=env,
                    capture_output=False,
                )

            if args.force or not gcc_edge_path.exists():
                extract_gcc_edge_list(edge_path, gcc_edge_path)

            for backend in selected_backends:
                for sample_count in [0, *sample_counts]:
                    sample_label: Any = "exact" if sample_count == 0 else sample_count
                    out_root = case_dir / f"embed_{backend}_{sample_label}"
                    coord_path = Path(f"{out_root}.inf_coord")
                    timing_path = Path(f"{out_root}.timing.json")

                    if args.force or not coord_path.exists() or not timing_path.exists():
                        timing = benchmark_embedding(
                            backend=backend,
                            embed_bin=embed_bins[backend],
                            edge_path=gcc_edge_path,
                            out_root=out_root,
                            dimension=dimension,
                            beta=beta,
                            seed=graph_seed,
                            negative_samples=sample_count,
                            env=env,
                        )
                        timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
                    else:
                        timing = json.loads(timing_path.read_text(encoding="utf-8"))

                    quality: Dict[str, Any] = {"objective": None, "c_score": None, "aligned_coord_corr": None}
                    if size <= args.quality_size_limit:
                        metrics_path = Path(f"{out_root}.metrics.json")
                        if args.force or not metrics_path.exists():
                            quality = compute_quality_metrics(gcc_edge_path, gen_coord_path, coord_path, dimension, args.block_size)
                            metrics_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
                        else:
                            quality = json.loads(metrics_path.read_text(encoding="utf-8"))

                    record = {
                        "dimension": dimension,
                        "size": size,
                        "backend": backend,
                        "sample_count": sample_label,
                        "negative_samples": sample_count,
                        "graph_seed": graph_seed,
                        "beta": beta,
                        "gamma": args.gamma,
                        "mean_degree": args.mean_degree,
                        "edge_file": str(gcc_edge_path),
                        "generated_coordinates": str(gen_coord_path),
                        "root": str(out_root),
                        "quality_enabled": size <= args.quality_size_limit,
                    }
                    for key in TIMING_KEYS:
                        record[key] = timing.get(key)
                    record["wall_time_ms"] = timing.get("wall_time_ms")
                    record["backend_reported"] = timing.get("backend")
                    record["mode"] = timing.get("mode")
                    record["refine_negative_samples_reported"] = timing.get("refine_negative_samples")
                    record.update(quality)
                    records.append(record)

    records.sort(key=lambda item: (item["dimension"], item["size"], item["backend"], -1 if item["sample_count"] == "exact" else item["sample_count"]))

    json_path = out_dir / "negative_sampling_cpu_gpu.json"
    csv_path = out_dir / "negative_sampling_cpu_gpu.csv"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    fieldnames = [
        "dimension",
        "size",
        "backend",
        "sample_count",
        "negative_samples",
        "graph_seed",
        "beta",
        "gamma",
        "mean_degree",
        "quality_enabled",
        "edge_file",
        "generated_coordinates",
        "root",
        "backend_reported",
        "mode",
        "refine_negative_samples_reported",
        "wall_time_ms",
        *TIMING_KEYS,
        "objective",
        "c_score",
        "aligned_coord_corr",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    plot_metric(records, dimensions, out_dir / "refine_time_vs_n.png", "refining_positions_ms", "Refinement Time vs System Size", "refining_positions_ms", y_scale="log")
    plot_metric(records, dimensions, out_dir / "total_time_vs_n.png", "total_time_ms", "Total Embedding Time vs System Size", "total_time_ms", y_scale="log")
    plot_metric(records, dimensions, out_dir / "wall_time_vs_n.png", "wall_time_ms", "Wall Time vs System Size", "wall_time_ms", y_scale="log")
    plot_metric(records, [1], out_dir / "c_score_vs_n_d1.png", "c_score", "C-score vs System Size", "c_score", include_only_with_metric=True)
    #plot_metric(records, [2], out_dir / "aligned_coord_corr_vs_n_d2.png", "aligned_coord_corr", "Aligned Coordinate Correlation vs System Size", "aligned_coord_corr", include_only_with_metric=True)
    #plot_metric(records, dimensions, out_dir / "objective_vs_n.png", "objective", "Exact Post-hoc Objective vs System Size", "objective", include_only_with_metric=True)

    print(f"Saved benchmark JSON: {json_path}")
    print(f"Saved benchmark CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
