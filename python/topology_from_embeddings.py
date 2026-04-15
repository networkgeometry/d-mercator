#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import ks_2samp


CASE_DIR_PATTERN = re.compile(r"^d(?P<dimension>\d+)_n(?P<size>\d+)_seed(?P<graph_seed>\d+)$")
EMBED_ROOT_PATTERN = re.compile(r"^embed_(?P<backend>cpu|gpu)_(?P<sample_count>exact|\d+)$")

TIMING_KEYS = [
    "total_time_ms",
    "initialization_ms",
    "parameter_inference_ms",
    "initial_positions_ms",
    "refining_positions_ms",
    "adjusting_kappas_ms",
    "io_ms",
]

SUMMARY_FIELDS = [
    "dimension",
    "size",
    "graph_seed",
    "backend",
    "sample_count",
    "negative_samples",
    "degree_ks",
    "avg_clustering_abs_diff",
    "transitivity_abs_diff",
    "ck_weighted_rmse",
    "spectrum_rel_rmse",
    "edge_count_abs_diff",
    "objective",
    "c_score",
    "aligned_coord_corr",
    "total_time_ms",
    "refining_positions_ms",
    "wall_time_ms",
    "mode",
    "backend_reported",
    "refine_negative_samples_reported",
    "root",
    "inf_coord_file",
    "edge_file",
    "synthetic_edge_file",
]

TOPOLOGY_METRICS = [
    ("degree_ks", "Degree KS"),
    ("avg_clustering_abs_diff", "Avg Clustering Abs Diff"),
    ("transitivity_abs_diff", "Transitivity Abs Diff"),
    ("ck_weighted_rmse", "c(k) Weighted RMSE"),
    ("spectrum_rel_rmse", "Spectrum Rel RMSE"),
    ("edge_count_abs_diff", "Edge Count Abs Diff"),
]

SAMPLE_COLORS = {
    "exact": "#1b9e77",
    "16": "#d95f02",
    "64": "#7570b3",
    "256": "#e7298a",
}
DEFAULT_SAMPLE_COLOR = "#4c78a8"
BACKEND_LINESTYLES = {"cpu": "-", "gpu": "--"}
BACKEND_MARKERS = {"cpu": "o", "gpu": "s"}
ORIGINAL_COLOR = "#111111"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate graphs from inferred embeddings and compare topology to the originals."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--benchmark-json", help="Path to negative_sampling_cpu_gpu.json")
    source_group.add_argument(
        "--job-dir",
        help="Benchmark job directory containing either artifacts/ or benchmark/artifacts/",
    )
    parser.add_argument("--generator-binary", required=True, help="Path to generate_sd")
    parser.add_argument("--output-dir", help="Destination for reports, caches, and plots")
    parser.add_argument("--size-limit", type=int, default=None, help="Optional upper bound on N")
    parser.add_argument("--spectral-rank", type=int, default=20, help="Number of leading eigenvalues to compare")
    parser.add_argument(
        "--min-degree-count",
        type=int,
        default=10,
        help="Minimum pooled node count required to draw a c(k) point",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate graphs and property caches")
    return parser.parse_args()


def run_cmd(cmd: Sequence[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(list(cmd), check=True)


def parse_case_directory_name(name: str) -> Dict[str, int]:
    match = CASE_DIR_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unrecognized case directory name: {name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def parse_embedding_root(path: Path) -> Dict[str, str]:
    stem = path.stem if path.suffix else path.name
    match = EMBED_ROOT_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"Unrecognized embedding filename: {path.name}")
    return {
        "backend": match.group("backend"),
        "sample_count": match.group("sample_count"),
    }


def sample_sort_key(sample_count: str) -> Tuple[int, int]:
    if sample_count == "exact":
        return (0, -1)
    return (1, int(sample_count))


def record_sort_key(record: Dict[str, Any]) -> Tuple[int, int, str, Tuple[int, int]]:
    return (
        int(record["dimension"]),
        int(record["size"]),
        str(record["backend"]),
        sample_sort_key(str(record["sample_count"])),
    )


def sample_color(sample_count: str) -> str:
    return SAMPLE_COLORS.get(sample_count, DEFAULT_SAMPLE_COLOR)


def dimension_plot_path(plots_dir: Path, base_name: str, dimension: int, dimensions: List[int]) -> Path:
    suffix = "" if len(dimensions) == 1 else f"_d{dimension}"
    return plots_dir / f"{base_name}{suffix}.png"


def maybe_float(value: Any) -> Any:
    if value is None:
        return None
    return float(value)


def load_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, name in enumerate(inferred["names"]):
            if dim == 1:
                handle.write(f"{name} {inferred['kappas'][index]:.17g} {inferred['theta'][index]:.17g}\n")
            else:
                coords = " ".join(f"{value:.17g}" for value in inferred["positions"][index])
                handle.write(f"{name} {inferred['kappas'][index]:.17g} {coords}\n")


def load_edges(edge_path: Path, name_to_index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
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
        "vertex_names": np.asarray(names),
        "degree": clustering["degree"],
        "clustering": clustering["clustering"],
        "average_clustering": clustering["average_clustering"],
        "transitivity": clustering["transitivity"],
        "spectrum": compute_spectrum(len(names), rows, cols, spectral_rank),
        "edge_count": len(rows) // 2,
    }


def save_properties(path: Path, properties: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertex_names=properties["vertex_names"],
        degree=properties["degree"],
        clustering=properties["clustering"],
        spectrum=properties["spectrum"],
        average_clustering=np.asarray([properties["average_clustering"]], dtype=float),
        transitivity=np.asarray([properties["transitivity"]], dtype=float),
        edge_count=np.asarray([properties["edge_count"]], dtype=np.int32),
    )


def load_properties(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {
        "vertex_names": data["vertex_names"],
        "degree": data["degree"],
        "clustering": data["clustering"],
        "spectrum": data["spectrum"],
        "average_clustering": float(data["average_clustering"][0]),
        "transitivity": float(data["transitivity"][0]),
        "edge_count": int(data["edge_count"][0]),
    }


def ensure_properties(edge_path: Path, names: List[str], cache_path: Path, spectral_rank: int, force: bool) -> Dict[str, Any]:
    if force or not cache_path.exists():
        save_properties(cache_path, compute_graph_properties(edge_path, names, spectral_rank))
    return load_properties(cache_path)


def compute_ck_summary(degree: np.ndarray, clustering: np.ndarray) -> Dict[int, Tuple[float, int]]:
    summary: Dict[int, List[float]] = {}
    for degree_value, clustering_value in zip(degree, clustering):
        entry = summary.setdefault(int(degree_value), [0.0, 0.0])
        entry[0] += float(clustering_value)
        entry[1] += 1
    return {degree_value: (total / count, int(count)) for degree_value, (total, count) in summary.items()}


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
    original_values = original["spectrum"][:rank]
    synthetic_values = synthetic["spectrum"][:rank]
    denominator = np.linalg.norm(original_values)
    if denominator == 0:
        return 0.0
    value = float(np.linalg.norm(original_values - synthetic_values) / denominator)
    if abs(value) < 1e-15:
        return 0.0
    return value


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
    sample_offset = 11 if str(record["sample_count"]) == "exact" else int(record["sample_count"])
    backend_offset = 0 if str(record["backend"]) == "cpu" else 1000
    return int(record["graph_seed"]) + backend_offset + sample_offset


def case_label(record: Dict[str, Any]) -> str:
    return f"d{record['dimension']}_n{record['size']}_seed{record['graph_seed']}"


def embedding_label(record: Dict[str, Any]) -> str:
    return f"{record['backend']}_{record['sample_count']}"


def normalize_benchmark_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    sample_count = str(raw_record["sample_count"])
    negative_samples = 0 if sample_count == "exact" else int(sample_count)
    record: Dict[str, Any] = {
        "dimension": int(raw_record["dimension"]),
        "size": int(raw_record["size"]),
        "graph_seed": int(raw_record["graph_seed"]),
        "backend": str(raw_record["backend"]),
        "sample_count": sample_count,
        "negative_samples": negative_samples,
        "root": str(Path(raw_record["root"]).resolve()),
        "inf_coord_file": str(Path(f"{raw_record['root']}.inf_coord").resolve()),
        "edge_file": str(Path(raw_record["edge_file"]).resolve()),
        "quality_enabled": bool(raw_record.get("quality_enabled", True)),
        "objective": raw_record.get("objective"),
        "c_score": raw_record.get("c_score"),
        "aligned_coord_corr": raw_record.get("aligned_coord_corr"),
        "wall_time_ms": raw_record.get("wall_time_ms"),
        "mode": raw_record.get("mode"),
        "backend_reported": raw_record.get("backend_reported"),
        "refine_negative_samples_reported": raw_record.get("refine_negative_samples_reported"),
    }
    for key in TIMING_KEYS:
        record[key] = raw_record.get(key)
    return record


def discover_artifacts_dir(job_dir: Path) -> Path:
    candidates = [job_dir / "artifacts", job_dir / "benchmark" / "artifacts"]
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) == 2:
        raise RuntimeError(
            f"Ambiguous job layout for {job_dir}: both {existing[0]} and {existing[1]} exist"
        )
    raise FileNotFoundError(f"Could not locate artifacts/ or benchmark/artifacts/ under {job_dir}")


def discover_job_records(job_dir: Path) -> List[Dict[str, Any]]:
    job_dir = job_dir.resolve()
    artifacts_dir = discover_artifacts_dir(job_dir)
    records: List[Dict[str, Any]] = []

    for case_dir in sorted(artifacts_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_info = parse_case_directory_name(case_dir.name)
        edge_file = case_dir / "synthetic_sd_GC.edge"
        if not edge_file.exists():
            raise FileNotFoundError(f"Missing original GCC edge list for {case_dir}: {edge_file}")

        for inf_coord_path in sorted(case_dir.glob("embed_*.inf_coord")):
            embed_info = parse_embedding_root(inf_coord_path)
            root = inf_coord_path.with_suffix("")
            timing_path = root.with_suffix(".timing.json")
            metrics_path = root.with_suffix(".metrics.json")

            timing = load_json_file(timing_path) if timing_path.exists() else {}
            quality = load_json_file(metrics_path) if metrics_path.exists() else {}
            sample_count = embed_info["sample_count"]
            negative_samples = 0 if sample_count == "exact" else int(sample_count)

            record: Dict[str, Any] = {
                "dimension": case_info["dimension"],
                "size": case_info["size"],
                "graph_seed": case_info["graph_seed"],
                "backend": embed_info["backend"],
                "sample_count": sample_count,
                "negative_samples": negative_samples,
                "root": str(root.resolve()),
                "inf_coord_file": str(inf_coord_path.resolve()),
                "edge_file": str(edge_file.resolve()),
                "quality_enabled": True,
                "objective": quality.get("objective"),
                "c_score": quality.get("c_score"),
                "aligned_coord_corr": quality.get("aligned_coord_corr"),
                "wall_time_ms": timing.get("wall_time_ms"),
                "mode": timing.get("mode"),
                "backend_reported": timing.get("backend"),
                "refine_negative_samples_reported": timing.get("refine_negative_samples"),
            }
            for key in TIMING_KEYS:
                record[key] = timing.get(key)
            records.append(record)

    records.sort(key=record_sort_key)
    return records


def load_input_records(args) -> Tuple[List[Dict[str, Any]], str]:
    if args.job_dir:
        job_dir = Path(args.job_dir).resolve()
        return discover_job_records(job_dir), str(job_dir)

    benchmark_json = Path(args.benchmark_json).resolve()
    raw_records = load_json_file(benchmark_json)
    records = [normalize_benchmark_record(raw_record) for raw_record in raw_records]
    records.sort(key=record_sort_key)
    return records, str(benchmark_json)


def resolve_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    if args.job_dir:
        return (Path(args.job_dir).resolve() / "topology_validation").resolve()
    return (Path(args.benchmark_json).resolve().parent / "topology").resolve()


def ensure_synthetic_graph(
    record: Dict[str, Any],
    inferred: Dict[str, Any],
    output_dir: Path,
    generator_binary: Path,
    force: bool,
) -> Path:
    synthetic_dir = output_dir / "synthetic_graphs" / case_label(record) / embedding_label(record)
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    hidden_vars_path = synthetic_dir / "synthetic_hidden_vars.txt"
    synthetic_root = synthetic_dir / "synthetic_from_embedding"
    synthetic_edge_path = synthetic_root.with_suffix(".edge")

    if force or not hidden_vars_path.exists():
        write_generator_hidden_vars(hidden_vars_path, inferred, int(record["dimension"]))

    if force or not synthetic_edge_path.exists():
        run_cmd(
            [
                str(generator_binary),
                "-d",
                str(record["dimension"]),
                "-n",
                "-t",
                "-b",
                str(inferred["beta"]),
                "-m",
                str(inferred["mu"]),
                "-s",
                str(synthetic_seed(record)),
                "-o",
                str(synthetic_root),
                str(hidden_vars_path),
            ]
        )

    return synthetic_edge_path


def process_record(
    record: Dict[str, Any],
    generator_binary: Path,
    output_dir: Path,
    spectral_rank: int,
    force: bool,
) -> Dict[str, Any]:
    inf_coord_path = Path(record["inf_coord_file"])
    inferred = parse_inferred_coordinates(inf_coord_path, int(record["dimension"]))
    synthetic_edge_path = ensure_synthetic_graph(record, inferred, output_dir, generator_binary, force)

    props_dir = output_dir / "properties" / case_label(record)
    original_cache = props_dir / "original.npz"
    synthetic_cache = props_dir / f"{embedding_label(record)}.npz"

    original_props = ensure_properties(Path(record["edge_file"]), inferred["names"], original_cache, spectral_rank, force)
    synthetic_props = ensure_properties(synthetic_edge_path, inferred["names"], synthetic_cache, spectral_rank, force)

    summary = dict(record)
    summary.update(compare_graphs(original_props, synthetic_props))
    summary["synthetic_edge_file"] = str(synthetic_edge_path.resolve())
    return {
        "summary": summary,
        "original_props": original_props,
        "synthetic_props": synthetic_props,
    }


def degree_ccdf(degrees: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    degree_values = np.sort(np.asarray(degrees, dtype=int))
    unique, counts = np.unique(degree_values, return_counts=True)
    survival = np.cumsum(counts[::-1])[::-1] / degree_values.size
    return unique, survival


def ck_curve(degrees: np.ndarray, clustering: np.ndarray, min_count: int) -> Tuple[np.ndarray, np.ndarray]:
    grouped: Dict[int, List[float]] = {}
    for degree_value, clustering_value in zip(degrees, clustering):
        entry = grouped.setdefault(int(degree_value), [0.0, 0.0])
        entry[0] += float(clustering_value)
        entry[1] += 1

    xs = []
    ys = []
    for degree_value in sorted(grouped):
        total, count = grouped[degree_value]
        if count < min_count:
            continue
        xs.append(degree_value)
        ys.append(total / count)
    return np.asarray(xs, dtype=int), np.asarray(ys, dtype=float)


def collect_legend(fig) -> None:
    handles = []
    labels = []
    seen = set()
    for axis in fig.axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))


def plot_metric_summary(summary_records: List[Dict[str, Any]], plots_dir: Path) -> None:
    if not summary_records:
        return

    dimensions = sorted({int(record["dimension"]) for record in summary_records})
    for dimension in dimensions:
        dim_records = [record for record in summary_records if int(record["dimension"]) == dimension]
        series_keys = sorted(
            {(str(record["backend"]), str(record["sample_count"])) for record in dim_records},
            key=lambda item: (item[0], sample_sort_key(item[1])),
        )

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
        for axis, (metric_key, title) in zip(axes.flat, TOPOLOGY_METRICS):
            for backend, sample_count in series_keys:
                series = sorted(
                    [
                        record
                        for record in dim_records
                        if str(record["backend"]) == backend and str(record["sample_count"]) == sample_count
                    ],
                    key=lambda item: int(item["size"]),
                )
                if not series:
                    continue
                axis.plot(
                    [int(record["size"]) for record in series],
                    [float(record[metric_key]) for record in series],
                    color=sample_color(sample_count),
                    linestyle=BACKEND_LINESTYLES.get(backend, "-"),
                    marker=BACKEND_MARKERS.get(backend, "o"),
                    linewidth=2,
                    label=f"{backend}/{sample_count}",
                )
            axis.set_title(title)
            axis.set_xlabel("N")
            axis.set_xscale("log")
            axis.grid(True, alpha=0.3)

        fig.suptitle(f"Topology Metrics vs System Size (D={dimension})", y=1.02)
        collect_legend(fig)
        fig.tight_layout()
        fig.savefig(
            dimension_plot_path(plots_dir, "topology_metrics_vs_n", dimension, dimensions),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_degree_ccdf(processed_records: List[Dict[str, Any]], plots_dir: Path) -> None:
    if not processed_records:
        return

    dimensions = sorted({int(item["summary"]["dimension"]) for item in processed_records})
    for dimension in dimensions:
        dim_records = [item for item in processed_records if int(item["summary"]["dimension"]) == dimension]
        sizes = sorted({int(item["summary"]["size"]) for item in dim_records})
        backends = sorted({str(item["summary"]["backend"]) for item in dim_records})
        fig, axes = plt.subplots(len(backends), len(sizes), figsize=(6 * len(sizes), 4 * len(backends)), squeeze=False)

        for row_index, backend in enumerate(backends):
            for col_index, size in enumerate(sizes):
                axis = axes[row_index][col_index]
                items = [
                    item
                    for item in dim_records
                    if str(item["summary"]["backend"]) == backend and int(item["summary"]["size"]) == size
                ]
                if not items:
                    axis.set_visible(False)
                    continue

                original_x, original_y = degree_ccdf(items[0]["original_props"]["degree"])
                axis.step(
                    original_x,
                    original_y,
                    where="post",
                    linewidth=2,
                    color=ORIGINAL_COLOR,
                    label="original",
                )

                for item in sorted(items, key=lambda value: sample_sort_key(str(value["summary"]["sample_count"]))):
                    sample_count = str(item["summary"]["sample_count"])
                    x_values, y_values = degree_ccdf(item["synthetic_props"]["degree"])
                    axis.step(
                        x_values,
                        y_values,
                        where="post",
                        linewidth=2,
                        color=sample_color(sample_count),
                        label=sample_count,
                    )

                axis.set_title(f"{backend.upper()} | N={size}")
                axis.set_xlabel("k")
                axis.set_ylabel("P(K >= k)")
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.grid(True, alpha=0.3)

        fig.suptitle(f"Degree CCDF: original vs synthetic from embeddings (D={dimension})", y=1.02)
        collect_legend(fig)
        fig.tight_layout()
        fig.savefig(
            dimension_plot_path(plots_dir, "degree_ccdf_vs_original", dimension, dimensions),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_ck(processed_records: List[Dict[str, Any]], plots_dir: Path, min_degree_count: int) -> None:
    if not processed_records:
        return

    dimensions = sorted({int(item["summary"]["dimension"]) for item in processed_records})
    for dimension in dimensions:
        dim_records = [item for item in processed_records if int(item["summary"]["dimension"]) == dimension]
        sizes = sorted({int(item["summary"]["size"]) for item in dim_records})
        backends = sorted({str(item["summary"]["backend"]) for item in dim_records})
        fig, axes = plt.subplots(len(backends), len(sizes), figsize=(6 * len(sizes), 4 * len(backends)), squeeze=False)

        for row_index, backend in enumerate(backends):
            for col_index, size in enumerate(sizes):
                axis = axes[row_index][col_index]
                items = [
                    item
                    for item in dim_records
                    if str(item["summary"]["backend"]) == backend and int(item["summary"]["size"]) == size
                ]
                if not items:
                    axis.set_visible(False)
                    continue

                original_x, original_y = ck_curve(
                    items[0]["original_props"]["degree"],
                    items[0]["original_props"]["clustering"],
                    min_degree_count,
                )
                axis.plot(original_x, original_y, linewidth=2, color=ORIGINAL_COLOR, label="original")

                for item in sorted(items, key=lambda value: sample_sort_key(str(value["summary"]["sample_count"]))):
                    sample_count = str(item["summary"]["sample_count"])
                    x_values, y_values = ck_curve(
                        item["synthetic_props"]["degree"],
                        item["synthetic_props"]["clustering"],
                        min_degree_count,
                    )
                    axis.plot(
                        x_values,
                        y_values,
                        linewidth=2,
                        color=sample_color(sample_count),
                        label=sample_count,
                    )

                axis.set_title(f"{backend.upper()} | N={size}")
                axis.set_xlabel("k")
                axis.set_ylabel("c(k)")
                axis.set_xscale("log")
                axis.set_ylim(0.0, 1.0)
                axis.grid(True, alpha=0.3)

        fig.suptitle(f"Clustering spectrum c(k): original vs synthetic from embeddings (D={dimension})", y=1.02)
        collect_legend(fig)
        fig.tight_layout()
        fig.savefig(
            dimension_plot_path(plots_dir, "clustering_spectrum_ck_vs_original", dimension, dimensions),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_spectrum(processed_records: List[Dict[str, Any]], plots_dir: Path) -> None:
    if not processed_records:
        return

    dimensions = sorted({int(item["summary"]["dimension"]) for item in processed_records})
    for dimension in dimensions:
        dim_records = [item for item in processed_records if int(item["summary"]["dimension"]) == dimension]
        sizes = sorted({int(item["summary"]["size"]) for item in dim_records})
        backends = sorted({str(item["summary"]["backend"]) for item in dim_records})
        fig, axes = plt.subplots(len(backends), len(sizes), figsize=(6 * len(sizes), 4 * len(backends)), squeeze=False)

        for row_index, backend in enumerate(backends):
            for col_index, size in enumerate(sizes):
                axis = axes[row_index][col_index]
                items = [
                    item
                    for item in dim_records
                    if str(item["summary"]["backend"]) == backend and int(item["summary"]["size"]) == size
                ]
                if not items:
                    axis.set_visible(False)
                    continue

                original_spectrum = items[0]["original_props"]["spectrum"]
                ranks = np.arange(1, len(original_spectrum) + 1)
                axis.plot(ranks, original_spectrum, linewidth=2, color=ORIGINAL_COLOR, marker="o", label="original")

                for item in sorted(items, key=lambda value: sample_sort_key(str(value["summary"]["sample_count"]))):
                    sample_count = str(item["summary"]["sample_count"])
                    synthetic_spectrum = item["synthetic_props"]["spectrum"]
                    axis.plot(
                        np.arange(1, len(synthetic_spectrum) + 1),
                        synthetic_spectrum,
                        linewidth=2,
                        color=sample_color(sample_count),
                        marker="o",
                        label=sample_count,
                    )

                axis.set_title(f"{backend.upper()} | N={size}")
                axis.set_xlabel("Eigenvalue Rank")
                axis.set_ylabel("Leading Adjacency Eigenvalue")
                axis.grid(True, alpha=0.3)

        fig.suptitle(f"Leading adjacency spectrum: original vs synthetic from embeddings (D={dimension})", y=1.02)
        collect_legend(fig)
        fig.tight_layout()
        fig.savefig(
            dimension_plot_path(plots_dir, "adjacency_spectrum_vs_original", dimension, dimensions),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def write_summary_csv(summary_records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in summary_records:
            writer.writerow({field: record.get(field) for field in SUMMARY_FIELDS})


def fmt_optional(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def write_report(summary_records: List[Dict[str, Any]], output_dir: Path, source_label: str) -> None:
    report_path = output_dir / "topology_from_embeddings_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Topology From Embeddings\n\n")
        handle.write(f"- Source: `{source_label}`\n")
        handle.write(f"- Records: {len(summary_records)}\n")
        handle.write("- Original graph reference: `synthetic_sd_GC.edge`\n")
        handle.write("- Plots: `plots/`\n\n")
        handle.write(
            "| D | N | Backend | Sample | Degree KS | Avg C Abs Diff | Transitivity Abs Diff | c(k) RMSE | Spectrum Rel RMSE | "
            "Edge Count Abs Diff | Objective | C-score | Wall Time (ms) |\n"
        )
        handle.write(
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for record in summary_records:
            handle.write(
                f"| {record['dimension']} | {record['size']} | {record['backend']} | {record['sample_count']} | "
                f"{fmt_optional(record['degree_ks'])} | {fmt_optional(record['avg_clustering_abs_diff'])} | "
                f"{fmt_optional(record['transitivity_abs_diff'])} | {fmt_optional(record['ck_weighted_rmse'])} | "
                f"{fmt_optional(record['spectrum_rel_rmse'])} | {fmt_optional(record['edge_count_abs_diff'])} | "
                f"{fmt_optional(record.get('objective'))} | {fmt_optional(record.get('c_score'))} | "
                f"{fmt_optional(record.get('wall_time_ms'))} |\n"
            )


def main() -> int:
    args = parse_args()
    generator_binary = Path(args.generator_binary).resolve()
    if not generator_binary.exists():
        raise SystemExit(f"Missing generator binary: {generator_binary}")

    records, source_label = load_input_records(args)
    if args.size_limit is not None:
        records = [record for record in records if int(record["size"]) <= args.size_limit]
    if not records:
        raise SystemExit("No embeddings found for topology validation.")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    processed_records = [
        process_record(record, generator_binary, output_dir, args.spectral_rank, args.force)
        for record in records
    ]
    summary_records = [item["summary"] for item in processed_records]
    summary_records.sort(key=record_sort_key)

    json_path = output_dir / "topology_from_embeddings.json"
    csv_path = output_dir / "topology_from_embeddings.csv"
    json_path.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
    write_summary_csv(summary_records, csv_path)
    write_report(summary_records, output_dir, source_label)

    plot_metric_summary(summary_records, plots_dir)
    plot_degree_ccdf(processed_records, plots_dir)
    plot_ck(processed_records, plots_dir, args.min_degree_count)
    plot_spectrum(processed_records, plots_dir)

    print(f"Saved topology JSON: {json_path}")
    print(f"Saved topology CSV: {csv_path}")
    print(f"Saved topology report: {output_dir / 'topology_from_embeddings_report.md'}")
    print(f"Saved plots directory: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
