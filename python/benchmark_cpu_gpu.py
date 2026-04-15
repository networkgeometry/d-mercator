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
from typing import Dict, List, Any

import networkx as nx


TIMING_KEYS = [
    "total_time_ms",
    "initialization_ms",
    "parameter_inference_ms",
    "initial_positions_ms",
    "initial_positions_weight_build_ms",
    "initial_positions_eigs_ms",
    "initial_positions_sort_ms",
    "initial_positions_gap_ms",
    "refining_positions_ms",
    "adjusting_kappas_ms",
    "io_ms",
]

DEFAULT_SIZES = [1000, 2000, 5000, 10000, 20000, 50000]


def parse_sizes(raw: str) -> List[int]:
    if not raw.strip():
        return list(DEFAULT_SIZES)
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_backends(raw: str) -> List[str]:
    if not raw.strip():
        raise SystemExit("--backends must not be empty")

    backends: List[str] = []
    for token in raw.split(","):
        backend = token.strip().lower()
        if not backend:
            continue
        if backend not in ("cpu", "gpu"):
            raise SystemExit(f"Unsupported backend in --backends: {backend}. Allowed: cpu,gpu")
        if backend not in backends:
            backends.append(backend)

    if not backends:
        raise SystemExit("--backends must include at least one of: cpu,gpu")
    return backends


def run_cmd(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    kwargs: Dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
        "universal_newlines": True,
        "check": True,
    }
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(cmd, **kwargs)


def parse_timing_json(stdout: str, stderr: str) -> Dict[str, Any]:
    combined_lines = stdout.splitlines() + stderr.splitlines()
    for line in reversed(combined_lines):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if "total_time_ms" in payload:
            return payload
    return {}


def write_hidden_variables(path: Path, nodes: int, dimension: int, seed: int) -> None:
    rng = random.Random(seed)
    gamma = 2.5
    mean_degree = 10.0
    kappa_0 = (
        (1.0 - 1.0 / nodes)
        / (1.0 - nodes ** ((2.0 - gamma) / (gamma - 1.0)))
        * (gamma - 2.0)
        / (gamma - 1.0)
        * mean_degree
    )
    kappa_c = kappa_0 * nodes ** (1.0 / (gamma - 1.0))
    scale = 1.0 - (kappa_c / kappa_0) ** (1.0 - gamma)

    with path.open("w", encoding="utf-8") as f:
        for _ in range(nodes):
            u = rng.random()
            kappa = kappa_0 * (1.0 - u * scale) ** (1.0 / (1.0 - gamma))
            if dimension == 1:
                theta = 2.0 * math.pi * rng.random()
                f.write(f"{kappa:.17g} {theta:.17g}\n")
                continue

            coords = [rng.gauss(0.0, 1.0) for _ in range(dimension + 1)]
            norm = math.sqrt(sum(c * c for c in coords))
            if norm == 0.0:
                coords[0] = 1.0
                norm = 1.0
            coords = [c / norm for c in coords]
            f.write(f"{kappa:.17g} {' '.join(f'{c:.17g}' for c in coords)}\n")


def extract_gcc_edge_list(edge_path: Path, out_path: Path) -> Path:
    graph = nx.read_edgelist(edge_path, comments="#", data=False)
    if graph.number_of_edges() == 0:
        raise RuntimeError(f"No edges found in {edge_path}")
    gcc_nodes = max(nx.connected_components(graph), key=len)
    gcc_graph = graph.subgraph(gcc_nodes).copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# source target\n")
        for u, v in gcc_graph.edges():
            f.write(f"{u} {v}\n")
    print(
        "GCC: kept"
        f" nodes={gcc_graph.number_of_nodes()}/{graph.number_of_nodes()}"
        f" edges={gcc_graph.number_of_edges()}/{graph.number_of_edges()}"
        f" -> {out_path}"
    )
    return out_path


def required_binary(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing executable: {path}")
    return path


def benchmark_backend(
    backend_name: str,
    embed_bin: Path,
    edge_path: Path,
    out_root: Path,
    dimension: int,
    beta: float,
    seed: int,
    env: Dict[str, str],
) -> Dict[str, Any]:
    cmd = [
        str(embed_bin),
        "-q",
        "-d",
        str(dimension),
        "-s",
        str(seed),
        "-b",
        str(beta),
        "--timing_json",
    ]
    if backend_name == "cpu":
        cmd.extend(["-C", "-D"])
    elif backend_name == "gpu":
        cmd.extend(["-G", "-D"])
    else:
        raise ValueError(f"Unsupported backend {backend_name}")
    cmd.extend(["-o", str(out_root), str(edge_path)])

    t0 = time.perf_counter()
    proc = run_cmd(cmd, cwd=edge_path.parent, env=env, capture_output=True)
    t1 = time.perf_counter()

    timing = parse_timing_json(proc.stdout, proc.stderr)
    if "total_time_ms" not in timing:
        print(
            f"WARNING: timing JSON not found for backend={backend_name} "
            f"at output root {out_root}; recording wall-time only."
        )
    timing["wall_time_ms"] = (t1 - t0) * 1000.0
    return timing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark embedding runtime on selected CPU/GPU backends for fixed network sizes."
    )
    parser.add_argument("--cpu-build-dir", help="Build directory configured with -DUSE_CUDA=OFF.")
    parser.add_argument("--gpu-build-dir", help="Build directory configured with -DUSE_CUDA=ON.")
    parser.add_argument(
        "--generator-build-dir",
        default=None,
        help="Build directory containing generate_sd. Defaults to cpu build dir when available, otherwise gpu build dir.",
    )
    parser.add_argument(
        "--backends",
        default="cpu,gpu",
        help="Comma-separated backends to benchmark (subset of: cpu,gpu). Default: cpu,gpu",
    )
    parser.add_argument("--out-dir", default="benchmarks/cpu_gpu")
    parser.add_argument("--sizes", default="1000,2000,5000,10000,20000,50000")
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--beta", type=float, default=4.5)
    parser.add_argument("--omp-threads", type=int, default=None)
    args = parser.parse_args()

    if args.dimension < 1:
        raise SystemExit("--dimension must be >= 1")
    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")

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

    out_dir = Path(args.out_dir).resolve()
    artifacts_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    generate_bin = required_binary(generator_build_dir / "generate_sd")
    embed_bins: Dict[str, Path] = {}
    if "cpu" in selected_backends:
        if cpu_build_dir is None:
            raise SystemExit("--cpu-build-dir is required when --backends includes cpu")
        embed_bins["cpu"] = required_binary(cpu_build_dir / "embed_sd")
    if "gpu" in selected_backends:
        if gpu_build_dir is None:
            raise SystemExit("--gpu-build-dir is required when --backends includes gpu")
        embed_bins["gpu"] = required_binary(gpu_build_dir / "embed_sd")

    sizes = parse_sizes(args.sizes)
    env = os.environ.copy()
    if args.omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(args.omp_threads)
    else:
        env.setdefault("OMP_NUM_THREADS", "1")

    records: List[Dict[str, Any]] = []

    for size in sizes:
        graph_seed = args.seed + args.dimension * 1_000_000 + size
        case_dir = artifacts_dir / f"d{args.dimension}_n{size}_seed{graph_seed}"
        case_dir.mkdir(parents=True, exist_ok=True)

        hidden_path = case_dir / "hidden_variables.txt"
        graph_root = case_dir / "synthetic_sd"
        edge_path = Path(str(graph_root) + ".edge")
        gcc_edge_path = Path(str(graph_root) + "_GC.edge")

        write_hidden_variables(hidden_path, size, args.dimension, graph_seed)

        t0_gen = time.perf_counter()
        run_cmd(
            [
                str(generate_bin),
                "-d",
                str(args.dimension),
                "-b",
                str(args.beta),
                "-s",
                str(graph_seed),
                "-o",
                str(graph_root),
                "-t",
                str(hidden_path),
            ],
            cwd=repo_root,
            env=env,
            capture_output=False,
        )
        t1_gen = time.perf_counter()
        graph_generation_wall_ms = (t1_gen - t0_gen) * 1000.0

        if not edge_path.exists():
            raise RuntimeError(f"Generated edge list missing: {edge_path}")

        gcc_edge_path = extract_gcc_edge_list(edge_path, gcc_edge_path)

        for rep in range(args.reps):
            rep_seed = graph_seed + rep
            for backend in selected_backends:
                embed_bin = embed_bins[backend]
                out_root = case_dir / f"embed_{backend}_rep{rep:02d}"
                timing = benchmark_backend(
                    backend_name=backend,
                    embed_bin=embed_bin,
                    edge_path=gcc_edge_path,
                    out_root=out_root,
                    dimension=args.dimension,
                    beta=args.beta,
                    seed=rep_seed,
                    env=env,
                )

                row: Dict[str, Any] = {
                    "size": size,
                    "rep": rep,
                    "dimension": args.dimension,
                    "graph_seed": graph_seed,
                    "embed_seed": rep_seed,
                    "beta": args.beta,
                    "graph_generation_wall_ms": graph_generation_wall_ms,
                    "wall_time_ms": float(timing["wall_time_ms"]),
                    "backend": backend,
                    "backend_reported": str(timing.get("backend", "")),
                    "mode": str(timing.get("mode", "")),
                    "nb_vertices_reported": int(timing.get("nb_vertices", size)),
                    "nb_edges_reported": int(timing.get("nb_edges", -1)),
                    "build_dir": str(embed_bin.parent),
                    "edge_file": str(gcc_edge_path),
                }
                for key in TIMING_KEYS:
                    row[key] = float(timing.get(key, float("nan")))
                records.append(row)

    csv_path = out_dir / "cpu_gpu_benchmark.csv"
    json_path = out_dir / "cpu_gpu_benchmark.json"

    fieldnames = [
        "size",
        "rep",
        "dimension",
        "graph_seed",
        "embed_seed",
        "beta",
        "backend",
        "backend_reported",
        "mode",
        "build_dir",
        "edge_file",
        "nb_vertices_reported",
        "nb_edges_reported",
        "graph_generation_wall_ms",
        "wall_time_ms",
    ] + TIMING_KEYS

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved benchmark CSV: {csv_path}")
    print(f"Saved benchmark JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
