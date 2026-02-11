#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from compare_coordinates import compare_coordinate_files


TIMING_KEYS = [
    "total_time_ms",
    "initialization_ms",
    "parameter_inference_ms",
    "initial_positions_ms",
    "refining_positions_ms",
    "adjusting_kappas_ms",
    "io_ms",
]

STACK_STAGES = [
    "initialization_ms",
    "parameter_inference_ms",
    "initial_positions_ms",
    "refining_positions_ms",
    "adjusting_kappas_ms",
    "io_ms",
]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_cmd(cmd: List[str], cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture_output,
        env=env,
    )


def parse_timing_json(stdout: str) -> Dict[str, float]:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if "total_time_ms" in parsed:
            return parsed
    raise RuntimeError("Timing JSON record not found in program output")


def write_hidden_variables(path: Path, nodes: int, dimension: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    kappas = rng.uniform(2.0, 12.0, size=nodes)
    with path.open("w", encoding="utf-8") as f:
        for i in range(nodes):
            if dimension == 1:
                theta = rng.uniform(0.0, 2.0 * np.pi)
                f.write(f"{kappas[i]:.17g} {theta:.17g}\n")
            else:
                vec = rng.normal(size=dimension + 1)
                norm = float(np.linalg.norm(vec))
                if norm == 0.0:
                    vec[0] = 1.0
                    norm = 1.0
                vec = vec / norm
                pos = " ".join(f"{x:.17g}" for x in vec)
                f.write(f"{kappas[i]:.17g} {pos}\n")


def iqr(values: List[float]) -> float:
    if not values:
        return float("nan")
    q75 = float(np.percentile(values, 75))
    q25 = float(np.percentile(values, 25))
    return q75 - q25


def summarize(records: List[Dict[str, float]], sizes: List[int], dimensions: List[int]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for d in dimensions:
        for n in sizes:
            row: Dict[str, float] = {
                "dimension": float(d),
                "nodes": float(n),
            }
            base = [r for r in records if int(r["dimension"]) == d and int(r["nodes"]) == n and r["mode"] == "baseline"]
            opt = [r for r in records if int(r["dimension"]) == d and int(r["nodes"]) == n and r["mode"] == "optimized"]

            if not base or not opt:
                continue

            for key in TIMING_KEYS:
                bvals = [float(r[key]) for r in base]
                ovals = [float(r[key]) for r in opt]
                bmed = float(np.median(bvals))
                omed = float(np.median(ovals))
                row[f"baseline_{key}_median"] = bmed
                row[f"optimized_{key}_median"] = omed
                row[f"baseline_{key}_iqr"] = iqr(bvals)
                row[f"optimized_{key}_iqr"] = iqr(ovals)
                row[f"speedup_{key}"] = (bmed / omed) if omed > 0 else float("nan")

            rows.append(row)

    rows.sort(key=lambda x: (int(x["dimension"]), int(x["nodes"])))
    return rows


def import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or run with --no-plots."
        ) from exc
    return plt


def plot_speedups(summary_rows: List[Dict[str, float]], out_dir: Path, dimensions: List[int]) -> None:
    plt = import_matplotlib_pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in dimensions:
        rows = [r for r in summary_rows if int(r["dimension"]) == d]
        if not rows:
            continue
        ns = [int(r["nodes"]) for r in rows]
        speedups = [float(r["speedup_total_time_ms"]) for r in rows]

        plt.figure(figsize=(7, 4.5))
        plt.plot(ns, speedups, marker="o", linewidth=2)
        plt.xlabel("N (nodes)")
        plt.ylabel("Speedup (baseline / optimized)")
        plt.title(f"Total Runtime Speedup vs N (d={d})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"speedup_d{d}.png", dpi=150)
        plt.close()


def plot_stage_stacks(summary_rows: List[Dict[str, float]], out_dir: Path, dimensions: List[int]) -> None:
    plt = import_matplotlib_pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]

    for d in dimensions:
        rows = [r for r in summary_rows if int(r["dimension"]) == d]
        if not rows:
            continue

        labels: List[str] = []
        values_by_stage = {s: [] for s in STACK_STAGES}
        for row in rows:
            n = int(row["nodes"])
            labels.extend([f"N={n}\nbase", f"N={n}\nopt"])
            for s in STACK_STAGES:
                values_by_stage[s].extend([
                    float(row[f"baseline_{s}_median"]),
                    float(row[f"optimized_{s}_median"]),
                ])

        x = np.arange(len(labels))
        bottom = np.zeros(len(labels), dtype=float)

        plt.figure(figsize=(max(8.0, len(labels) * 0.8), 5.0))
        for stage, color in zip(STACK_STAGES, colors):
            vals = np.array(values_by_stage[stage], dtype=float)
            plt.bar(x, vals, bottom=bottom, label=stage.replace("_ms", ""), color=color)
            bottom += vals

        plt.xticks(x, labels)
        plt.ylabel("Median stage time (ms)")
        plt.title(f"Stage Time Breakdown: baseline vs optimized (d={d})")
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout()
        plt.savefig(out_dir / f"stage_breakdown_d{d}.png", dpi=150)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline-vs-optimized embedding benchmarks with equivalence checks.")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--out", default="bench")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--sizes", default="500,2000,8000")
    parser.add_argument("--dimensions", default="1,2")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--beta", type=float, default=4.5)
    parser.add_argument("--rmse-tol", type=float, default=1e-8)
    parser.add_argument("--max-abs-tol", type=float, default=1e-6)
    parser.add_argument("--pair-samples", type=int, default=20000)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip Matplotlib plot generation.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = (repo_root / args.build_dir).resolve()
    out_dir = (repo_root / args.out).resolve()
    artifacts_dir = out_dir / "artifacts"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sizes = parse_int_list(args.sizes)
    dimensions = parse_int_list(args.dimensions)
    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")

    if not args.skip_build:
        run_cmd([
            "cmake", "-S", str(repo_root), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release", "-DDMERCATOR_RELEASE_O3=ON"
        ], repo_root)
        run_cmd(["cmake", "--build", str(build_dir), "-j"], repo_root)

    generate_bin = build_dir / "generate_sd"
    embed_bin = build_dir / "embed_sd"
    if not generate_bin.exists():
        raise SystemExit(f"Missing generator executable: {generate_bin}")
    if not embed_bin.exists():
        raise SystemExit(f"Missing embedder executable: {embed_bin}")

    timing_records: List[Dict[str, float]] = []
    comparison_results: List[Dict[str, object]] = []

    for d in dimensions:
        for n in sizes:
            cfg_seed = args.seed + d * 1_000_000 + n
            cfg_dir = artifacts_dir / f"d{d}_n{n}_seed{cfg_seed}"
            cfg_dir.mkdir(parents=True, exist_ok=True)

            hidden_path = cfg_dir / "hidden_variables.txt"
            write_hidden_variables(hidden_path, n, d, cfg_seed)

            graph_root = cfg_dir / "synthetic_sd"
            run_cmd([
                str(generate_bin),
                "-d", str(d),
                "-b", str(args.beta),
                "-s", str(cfg_seed),
                "-o", str(graph_root),
                "-t", str(hidden_path),
            ], repo_root)

            edge_path = Path(str(graph_root) + ".edge")
            truth_csv = Path(str(graph_root) + ".truth.csv")
            if not edge_path.exists():
                raise RuntimeError(f"Generated edge list missing: {edge_path}")

            mode_first_csv: Dict[str, Path] = {}
            for mode in ("baseline", "optimized"):
                for rep in range(args.reps):
                    out_root = cfg_dir / f"embed_{mode}_rep{rep:02d}"
                    cmd = [
                        str(embed_bin),
                        "-q",
                        "-d", str(d),
                        "-s", str(cfg_seed),
                        "-b", str(args.beta),
                        "--mode", mode,
                        "--timing_json",
                        "-o", str(out_root),
                        str(edge_path),
                    ]
                    proc = run_cmd(cmd, repo_root, capture_output=True)
                    timing = parse_timing_json(proc.stdout)
                    timing_record: Dict[str, float] = {
                        "dimension": float(d),
                        "nodes": float(n),
                        "seed": float(cfg_seed),
                        "rep": float(rep),
                        "mode": mode,
                    }
                    for key in TIMING_KEYS:
                        timing_record[key] = float(timing[key])
                    timing_records.append(timing_record)

                    coords_csv = Path(str(out_root) + ".coords.csv")
                    if not coords_csv.exists():
                        raise RuntimeError(f"Expected coordinates CSV not found: {coords_csv}")
                    if rep == 0:
                        mode_first_csv[mode] = coords_csv

            compare_debug_dir = cfg_dir / "comparison_debug"
            compare_result = compare_coordinate_files(
                baseline_csv=mode_first_csv["baseline"],
                optimized_csv=mode_first_csv["optimized"],
                dimension=d,
                rmse_tol=args.rmse_tol,
                max_abs_tol=args.max_abs_tol,
                pair_samples=args.pair_samples,
                seed=cfg_seed,
                truth_csv=truth_csv if truth_csv.exists() else None,
                debug_dir=compare_debug_dir,
            )
            compare_result["dimension"] = d
            compare_result["nodes"] = n
            compare_result["seed"] = cfg_seed
            compare_result["baseline_csv"] = str(mode_first_csv["baseline"])
            compare_result["optimized_csv"] = str(mode_first_csv["optimized"])
            comparison_results.append(compare_result)

            if not compare_result["passed"]:
                summary_path = cfg_dir / "comparison_failed.json"
                summary_path.write_text(json.dumps(compare_result, indent=2), encoding="utf-8")
                print(f"Coordinate equivalence FAILED for d={d}, N={n}. Details: {summary_path}")
                return 2

    summary_rows = summarize(timing_records, sizes=sizes, dimensions=dimensions)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["dimension"] = summary_df["dimension"].astype(int)
        summary_df["nodes"] = summary_df["nodes"].astype(int)

    results_csv = out_dir / "results.csv"
    summary_df.to_csv(results_csv, index=False)

    raw_timing_path = out_dir / "timing_raw.json"
    raw_timing_path.write_text(json.dumps(timing_records, indent=2), encoding="utf-8")

    compare_path = out_dir / "equivalence_results.json"
    compare_path.write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")

    full_json = {
        "config": {
            "sizes": sizes,
            "dimensions": dimensions,
            "reps": args.reps,
            "seed": args.seed,
            "beta": args.beta,
            "rmse_tol": args.rmse_tol,
            "max_abs_tol": args.max_abs_tol,
            "pair_samples": args.pair_samples,
        },
        "summary": summary_rows,
        "timing_raw": timing_records,
        "equivalence": comparison_results,
    }
    results_json = out_dir / "results.json"
    results_json.write_text(json.dumps(full_json, indent=2), encoding="utf-8")

    if not args.no_plots:
        plot_speedups(summary_rows, plots_dir, dimensions=dimensions)
        plot_stage_stacks(summary_rows, plots_dir, dimensions=dimensions)

    print("Benchmark completed successfully")
    print(f"  Summary CSV: {results_csv}")
    print(f"  Summary JSON: {results_json}")
    print(f"  Equivalence: {compare_path}")
    if args.no_plots:
        print("  Plots dir: skipped (--no-plots)")
    else:
        print(f"  Plots dir: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
