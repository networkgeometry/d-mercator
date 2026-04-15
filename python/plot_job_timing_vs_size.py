#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from topology_from_embeddings import (
    BACKEND_LINESTYLES,
    BACKEND_MARKERS,
    DEFAULT_SAMPLE_COLOR,
    SAMPLE_COLORS,
    discover_artifacts_dir,
    parse_case_directory_name,
    parse_embedding_root,
    sample_sort_key,
)


TIMING_METRICS: List[Tuple[str, str, str]] = [
    ("total_time_ms", "Total Embedding Time vs System Size", "Total time"),
    ("wall_time_ms", "Wall Time vs System Size", "Wall time"),
    ("refining_positions_ms", "Refinement Time vs System Size", "Refinement time"),
]

COMPLEXITY_MODELS: List[Tuple[str, str, str]] = [
    ("n_log_n", "O(N log N)", ":"),
    ("n_squared", "O(N^2)", "-."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot embedding time metrics versus system size for an existing benchmark job folder."
    )
    parser.add_argument("--job-dir", required=True, help="Benchmark job directory containing artifacts/")
    parser.add_argument(
        "--output-dir",
        help="Directory for plot outputs. Defaults to <job-dir>/topology_validation/plots",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Optional dimension filter, e.g. 1",
    )
    parser.add_argument(
        "--time-unit",
        choices=("ms", "minutes"),
        default="ms",
        help="Unit to use on the y-axis and in output filenames.",
    )
    parser.add_argument(
        "--add-complexity-fits",
        action="store_true",
        help="Write extra plots with per-series fitted O(N log N) and O(N^2) reference curves.",
    )
    return parser.parse_args()


def time_unit_scale(time_unit: str) -> float:
    if time_unit == "minutes":
        return 1.0 / (1000.0 * 60.0)
    return 1.0


def time_unit_suffix(time_unit: str) -> str:
    return "" if time_unit == "ms" else f"_{time_unit}"


def y_axis_label(metric_label: str, time_unit: str) -> str:
    unit_label = "ms" if time_unit == "ms" else "minutes"
    return f"{metric_label} ({unit_label})"


def sorted_series_labels(records: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        {series_label(record) for record in records},
        key=lambda value: (
            value.split("/")[0],
            0 if value.endswith("/exact") else 1,
            int(value.split("=")[-1]) if "S=" in value else -1,
        ),
    )


def complexity_basis(model_key: str, sizes: List[int]) -> List[float]:
    if model_key == "n_squared":
        return [float(size) ** 2 for size in sizes]
    if model_key == "n_log_n":
        return [float(size) * math.log(float(size)) for size in sizes]
    raise ValueError(f"Unsupported complexity model: {model_key}")


def fit_scale_in_log_space(observed: List[float], basis: List[float]) -> float:
    log_scale = sum(math.log(obs) - math.log(base) for obs, base in zip(observed, basis)) / len(observed)
    return math.exp(log_scale)


def series_label(record: Dict[str, Any]) -> str:
    sample_count = str(record["sample_count"])
    sample_label = "exact" if sample_count == "exact" else f"S={sample_count}"
    return f"{record['backend']}/{sample_label}"


def series_style(record: Dict[str, Any]) -> Dict[str, Any]:
    sample_count = str(record["sample_count"])
    backend = str(record["backend"])
    return {
        "color": SAMPLE_COLORS.get(sample_count, DEFAULT_SAMPLE_COLOR),
        "linestyle": BACKEND_LINESTYLES.get(backend, "-"),
        "marker": BACKEND_MARKERS.get(backend, "o"),
    }


def discover_timing_records(job_dir: Path, dimension: int | None) -> List[Dict[str, Any]]:
    artifacts_dir = discover_artifacts_dir(job_dir)
    records: List[Dict[str, Any]] = []

    for case_dir in sorted(artifacts_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_info = parse_case_directory_name(case_dir.name)
        if dimension is not None and case_info["dimension"] != dimension:
            continue

        for timing_path in sorted(case_dir.glob("embed_*.timing.json")):
            embed_info = parse_embedding_root(timing_path.with_suffix(""))
            payload = json.loads(timing_path.read_text(encoding="utf-8"))
            record: Dict[str, Any] = {
                "dimension": case_info["dimension"],
                "size": case_info["size"],
                "graph_seed": case_info["graph_seed"],
                "backend": embed_info["backend"],
                "sample_count": str(embed_info["sample_count"]),
                "nb_vertices": payload.get("nb_vertices"),
                "total_time_ms": payload.get("total_time_ms"),
                "wall_time_ms": payload.get("wall_time_ms"),
                "refining_positions_ms": payload.get("refining_positions_ms"),
                "timing_file": str(timing_path.resolve()),
            }
            records.append(record)

    records.sort(
        key=lambda item: (
            int(item["dimension"]),
            int(item["size"]),
            str(item["backend"]),
            sample_sort_key(str(item["sample_count"])),
        )
    )
    return records


def plot_metric(
    records: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    metric_label: str,
    time_unit: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5.5))
    scale = time_unit_scale(time_unit)

    labels = sorted_series_labels(records)

    for label in labels:
        series = sorted(
            [record for record in records if series_label(record) == label and record.get(metric_key) is not None],
            key=lambda item: int(item["size"]),
        )
        if not series:
            continue
        style = series_style(series[0])
        axis.plot(
            [record["size"] for record in series],
            [float(record[metric_key]) * scale for record in series],
            label=label,
            linewidth=2.0,
            markersize=7,
            **style,
        )

    axis.set_title(title)
    axis.set_xlabel("System size N")
    axis.set_ylabel(y_axis_label(metric_label, time_unit))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_with_complexity_fits(
    records: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    metric_label: str,
    time_unit: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 6.0))
    scale = time_unit_scale(time_unit)
    labels = sorted_series_labels(records)

    for label in labels:
        series = sorted(
            [record for record in records if series_label(record) == label and record.get(metric_key) is not None],
            key=lambda item: int(item["size"]),
        )
        if not series:
            continue

        style = series_style(series[0])
        sizes = [int(record["size"]) for record in series]
        observed = [float(record[metric_key]) * scale for record in series]
        axis.plot(
            sizes,
            observed,
            label=label,
            linewidth=2.2,
            markersize=7,
            **style,
        )

        if len(series) < 2 or any(value <= 0.0 for value in observed):
            continue

        for model_key, _, linestyle in COMPLEXITY_MODELS:
            basis = complexity_basis(model_key, sizes)
            fitted_scale = fit_scale_in_log_space(observed, basis)
            fitted = [fitted_scale * value for value in basis]
            axis.plot(
                sizes,
                fitted,
                color=style["color"],
                linestyle=linestyle,
                linewidth=1.8,
                alpha=0.8,
            )

    axis.set_title(f"{title} with Complexity Fits")
    axis.set_xlabel("System size N")
    axis.set_ylabel(y_axis_label(metric_label, time_unit))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)

    series_legend = axis.legend(loc="upper left", title="Series")
    axis.add_artist(series_legend)
    fit_handles = [
        Line2D([0], [0], color="#555555", linestyle=linestyle, linewidth=1.8, label=model_label)
        for _, model_label, linestyle in COMPLEXITY_MODELS
    ]
    axis.legend(handles=fit_handles, loc="lower right", title="Reference fits")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(records: List[Dict[str, Any]], time_unit: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(TIMING_METRICS), figsize=(5.5 * len(TIMING_METRICS), 5.0), squeeze=False)
    scale = time_unit_scale(time_unit)

    labels = sorted_series_labels(records)

    for axis, (metric_key, title, metric_label) in zip(axes[0], TIMING_METRICS):
        for label in labels:
            series = sorted(
                [record for record in records if series_label(record) == label and record.get(metric_key) is not None],
                key=lambda item: int(item["size"]),
            )
            if not series:
                continue
            style = series_style(series[0])
            axis.plot(
                [record["size"] for record in series],
                [float(record[metric_key]) * scale for record in series],
                label=label,
                linewidth=2.0,
                markersize=6,
                **style,
            )
        axis.set_title(title)
        axis.set_xlabel("System size N")
        axis.set_ylabel(y_axis_label(metric_label, time_unit))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    job_dir = Path(args.job_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (job_dir / "topology_validation" / "plots").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_timing_records(job_dir, args.dimension)
    if not records:
        raise SystemExit(f"No timing records found under {job_dir}")

    output_suffix = time_unit_suffix(args.time_unit)
    for metric_key, title, metric_label in TIMING_METRICS:
        plot_metric(
            records,
            metric_key,
            title,
            metric_label,
            args.time_unit,
            output_dir / f"{metric_key}{output_suffix}_vs_n.png",
        )
        if args.add_complexity_fits:
            plot_metric_with_complexity_fits(
                records,
                metric_key,
                title,
                metric_label,
                args.time_unit,
                output_dir / f"{metric_key}{output_suffix}_vs_n_complexity_fits.png",
            )
    plot_dashboard(records, args.time_unit, output_dir / f"timing{output_suffix}_vs_n.png")

    print(f"Saved timing plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
