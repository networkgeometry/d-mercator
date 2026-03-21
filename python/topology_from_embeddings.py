#!/usr/bin/env python3

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import ks_2samp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate graphs from inferred embeddings and compare topology to the originals."
    )
    parser.add_argument("--benchmark-json", required=True, help="Path to negative_sampling_cpu_gpu.json")
    parser.add_argument("--generator-binary", required=True, help="Path to generate_sd")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size-limit", type=int, default=10000)
    parser.add_argument("--spectral-rank", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
        "kappas": kappas,
        "theta": theta,
        "positions": positions,
    }


def write_generator_hidden_vars(path: Path, inferred: Dict[str, Any], dim: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, name in enumerate(inferred["names"]):
            if dim == 1:
                handle.write(f"{name} {inferred['kappas'][index]:.17g} {inferred['theta'][index]:.17g}\n")
            else:
                coords = " ".join(f"{value:.17g}" for value in inferred["positions"][index])
                handle.write(f"{name} {inferred['kappas'][index]:.17g} {coords}\n")


def load_edges(edge_path: Path, name_to_index: Dict[str, int]) -> np.ndarray:
    rows = []
    cols = []
    with edge_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            source, target = stripped.split()[:2]
            if source == target:
                continue
            v1 = name_to_index[source]
            v2 = name_to_index[target]
            rows.extend((v1, v2))
            cols.extend((v2, v1))
    return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)


def build_adjacency_lists(size: int, rows: np.ndarray, cols: np.ndarray) -> List[set]:
    adjacency = [set() for _ in range(size)]
    for v1, v2 in zip(rows, cols):
        adjacency[v1].add(v2)
    return adjacency


def compute_clustering_metrics(adjacency: List[set]) -> Dict[str, Any]:
    size = len(adjacency)
    degree = np.fromiter((len(neighbors) for neighbors in adjacency), dtype=np.int32, count=size)
    clustering = np.zeros(size, dtype=float)
    closed_triplets = 0
    total_triplets = 0

    for vertex, neighbors in enumerate(adjacency):
        d = degree[vertex]
        if d < 2:
            continue
        neighbors_list = list(neighbors)
        total_triplets += d * (d - 1) // 2
        triangles_at_vertex = 0
        for index, neighbor in enumerate(neighbors_list):
            neighbor_set = adjacency[neighbor]
            for other in neighbors_list[index + 1 :]:
                if other in neighbor_set:
                    triangles_at_vertex += 1
        clustering[vertex] = (2.0 * triangles_at_vertex) / (d * (d - 1))
        closed_triplets += triangles_at_vertex

    average_clustering = float(clustering.mean())
    transitivity = float(closed_triplets / total_triplets) if total_triplets else 0.0
    return {
        "degree": degree,
        "clustering": clustering,
        "average_clustering": average_clustering,
        "transitivity": transitivity,
    }


def compute_spectrum(size: int, rows: np.ndarray, cols: np.ndarray, spectral_rank: int) -> np.ndarray:
    max_rank = min(spectral_rank, max(1, size - 2))
    adjacency = csr_matrix((np.ones_like(rows, dtype=float), (rows, cols)), shape=(size, size))
    if adjacency.nnz == 0:
        return np.zeros(max_rank, dtype=float)
    eigenvalues = eigsh(adjacency, k=max_rank, which="LA", return_eigenvectors=False, tol=1e-3)
    return np.sort(eigenvalues)[::-1]


def compute_graph_properties(edge_path: Path, names: List[str], spectral_rank: int) -> Dict[str, Any]:
    name_to_index = {name: index for index, name in enumerate(names)}
    rows, cols = load_edges(edge_path, name_to_index)
    adjacency = build_adjacency_lists(len(names), rows, cols)
    clustering = compute_clustering_metrics(adjacency)
    return {
        "degree": clustering["degree"],
        "clustering": clustering["clustering"],
        "average_clustering": clustering["average_clustering"],
        "transitivity": clustering["transitivity"],
        "spectrum": compute_spectrum(len(names), rows, cols, spectral_rank),
        "edge_count": len(rows) // 2,
    }


def compute_ck_summary(degree: np.ndarray, clustering: np.ndarray) -> Dict[int, Any]:
    summary: Dict[int, Any] = {}
    for degree_value, clustering_value in zip(degree, clustering):
        total, count = summary.setdefault(int(degree_value), [0.0, 0])
        summary[int(degree_value)] = [total + float(clustering_value), count + 1]
    return {degree_value: (total / count, count) for degree_value, (total, count) in summary.items()}


def compute_ck_weighted_rmse(original: Dict[str, Any], synthetic: Dict[str, Any]) -> float:
    original_ck = compute_ck_summary(original["degree"], original["clustering"])
    synthetic_ck = compute_ck_summary(synthetic["degree"], synthetic["clustering"])
    weighted_squared_error = 0.0
    total_weight = 0
    for degree_value, (original_mean, count) in original_ck.items():
        synthetic_mean = synthetic_ck.get(degree_value, (0.0, 0))[0]
        weighted_squared_error += count * (original_mean - synthetic_mean) ** 2
        total_weight += count
    return math.sqrt(weighted_squared_error / total_weight) if total_weight else 0.0


def compute_spectrum_relative_rmse(original: Dict[str, Any], synthetic: Dict[str, Any]) -> float:
    rank = min(len(original["spectrum"]), len(synthetic["spectrum"]))
    if rank == 0:
        return 0.0
    numerator = np.linalg.norm(original["spectrum"][:rank] - synthetic["spectrum"][:rank])
    denominator = np.linalg.norm(original["spectrum"][:rank])
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compare_graphs(original: Dict[str, Any], synthetic: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "degree_ks": float(ks_2samp(original["degree"], synthetic["degree"]).statistic),
        "avg_clustering_abs_diff": abs(original["average_clustering"] - synthetic["average_clustering"]),
        "transitivity_abs_diff": abs(original["transitivity"] - synthetic["transitivity"]),
        "ck_weighted_rmse": compute_ck_weighted_rmse(original, synthetic),
        "spectrum_rel_rmse": compute_spectrum_relative_rmse(original, synthetic),
        "edge_count_abs_diff": abs(original["edge_count"] - synthetic["edge_count"]),
    }


def synthetic_seed(record: Dict[str, Any]) -> int:
    sample_offset = 11 if record["sample_count"] == "exact" else int(record["sample_count"])
    backend_offset = 0 if record["backend"] == "cpu" else 1000
    return record["graph_seed"] + backend_offset + sample_offset


def main() -> int:
    args = parse_args()
    records = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_records: List[Dict[str, Any]] = []
    for record in records:
        if record["size"] > args.size_limit or not record["quality_enabled"]:
            continue

        dim = int(record["dimension"])
        root = Path(record["root"])
        inferred = parse_inferred_coordinates(Path(f"{root}.inf_coord"), dim)
        case_dir = output_dir / f"d{dim}_n{record['size']}" / record["backend"] / str(record["sample_count"])
        case_dir.mkdir(parents=True, exist_ok=True)
        hidden_vars_path = case_dir / "synthetic_hidden_vars.txt"
        synthetic_root = case_dir / "synthetic_from_embedding"
        synthetic_edge_path = Path(f"{synthetic_root}.edge")

        if args.force or not synthetic_edge_path.exists():
            write_generator_hidden_vars(hidden_vars_path, inferred, dim)
            run_cmd(
                [
                    args.generator_binary,
                    "-d", str(dim),
                    "-n",
                    "-t",
                    "-b", str(inferred["beta"]),
                    "-m", str(inferred["mu"]),
                    "-s", str(synthetic_seed(record)),
                    "-o", str(synthetic_root),
                    str(hidden_vars_path),
                ]
            )

        original = compute_graph_properties(Path(record["edge_file"]), inferred["names"], args.spectral_rank)
        synthetic = compute_graph_properties(synthetic_edge_path, inferred["names"], args.spectral_rank)
        comparison = compare_graphs(original, synthetic)
        comparison.update(
            {
                "dimension": record["dimension"],
                "size": record["size"],
                "backend": record["backend"],
                "sample_count": record["sample_count"],
                "root": record["root"],
                "synthetic_edge_file": str(synthetic_edge_path),
            }
        )
        topology_records.append(comparison)

    topology_records.sort(
        key=lambda item: (item["dimension"], item["size"], item["backend"], -1 if item["sample_count"] == "exact" else item["sample_count"])
    )

    json_path = output_dir / "topology_from_embeddings.json"
    csv_path = output_dir / "topology_from_embeddings.csv"
    json_path.write_text(json.dumps(topology_records, indent=2), encoding="utf-8")

    fieldnames = [
        "dimension",
        "size",
        "backend",
        "sample_count",
        "degree_ks",
        "avg_clustering_abs_diff",
        "transitivity_abs_diff",
        "ck_weighted_rmse",
        "spectrum_rel_rmse",
        "edge_count_abs_diff",
        "root",
        "synthetic_edge_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in topology_records:
            writer.writerow(record)

    report_path = output_dir / "topology_from_embeddings_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Topology From Embeddings\n\n")
        handle.write("| D | N | Backend | Sample | Degree KS | Avg C Abs Diff | Transitivity Abs Diff | c(k) RMSE | Spectrum Rel RMSE | Edge Count Abs Diff |\n")
        handle.write("| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for record in topology_records:
            handle.write(
                f"| {record['dimension']} | {record['size']} | {record['backend']} | {record['sample_count']} | "
                f"{record['degree_ks']:.6f} | {record['avg_clustering_abs_diff']:.6f} | "
                f"{record['transitivity_abs_diff']:.6f} | {record['ck_weighted_rmse']:.6f} | "
                f"{record['spectrum_rel_rmse']:.6f} | {record['edge_count_abs_diff']} |\n"
            )

    print(f"Saved topology JSON: {json_path}")
    print(f"Saved topology CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
