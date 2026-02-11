#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Install numpy and matplotlib.")

def load_legacy(path: Path, dimension: int):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if dimension == 1:
                if len(parts) < 4:
                    continue
                node = parts[0]
                rows[node] = {
                    "kappa": float(parts[1]),
                    "theta_0": float(parts[2]),
                    "r": float(parts[3]),
                }
            else:
                needed = 3 + (dimension + 1)
                if len(parts) < needed:
                    continue
                node = parts[0]
                row = {
                    "kappa": float(parts[1]),
                    "r": float(parts[2]),
                }
                for i in range(dimension + 1):
                    row[f"pos_{i}"] = float(parts[3 + i])
                rows[node] = row
    return rows

def load_csv(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row["node_id"]
            parsed = {}
            for key, value in row.items():
                if key == "node_id" or value is None or value == "":
                    continue
                parsed[key] = float(value)
            rows[node] = parsed
    return rows

def load_table(path: Path, dimension: int):
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    return load_legacy(path, dimension)

def circular_diff(a, b):
    return np.angle(np.exp(1j * (a - b)))

def rmse(diff):
    return float(np.sqrt(np.mean(diff * diff)))

def max_abs(diff):
    return float(np.max(np.abs(diff)))

def component_names(dimension: int):
    names = ["kappa", "r"]
    if dimension == 1:
        names.append("theta_0")
    else:
        for i in range(dimension + 1):
            names.append(f"pos_{i}")
    return names

def main():
    parser = argparse.ArgumentParser(description="Compare two embedding outputs with deterministic node matching.")
    parser.add_argument("--old", required=True, help="Old/baseline output (.inf_coord or .csv)")
    parser.add_argument("--new", required=True, help="New/refactored output (.inf_coord or .csv)")
    parser.add_argument("--dimension", type=int, required=True, help="Embedding dimension")
    parser.add_argument("--output-dir", default="validation_output/compare", help="Directory for plots")
    parser.add_argument("--plot-component", default=None, help="Component to plot (default: theta_0 or pos_0)")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_rows = load_table(old_path, args.dimension)
    new_rows = load_table(new_path, args.dimension)

    common_nodes = sorted(set(old_rows.keys()) & set(new_rows.keys()))
    if not common_nodes:
        raise RuntimeError("No common node ids found between old and new outputs")

    comps = component_names(args.dimension)
    metrics = {}

    for comp in comps:
        old_vals = np.array([old_rows[n][comp] for n in common_nodes], dtype=float)
        new_vals = np.array([new_rows[n][comp] for n in common_nodes], dtype=float)
        if comp == "theta_0":
            diff = circular_diff(new_vals, old_vals)
        else:
            diff = new_vals - old_vals
        metrics[comp] = {
            "rmse": rmse(diff),
            "max_abs": max_abs(diff),
        }

    print(f"Compared nodes: {len(common_nodes)}")
    for comp in comps:
        m = metrics[comp]
        print(f"{comp:10s} rmse={m['rmse']:.12g} max_abs={m['max_abs']:.12g}")

    plot_comp = args.plot_component
    if plot_comp is None:
        plot_comp = "theta_0" if args.dimension == 1 else "pos_0"
    if plot_comp not in comps:
        raise RuntimeError(f"plot component '{plot_comp}' not found in components {comps}")

    x = np.array([old_rows[n][plot_comp] for n in common_nodes], dtype=float)
    y = np.array([new_rows[n][plot_comp] for n in common_nodes], dtype=float)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=10, alpha=0.7)
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel(f"old {plot_comp}")
    plt.ylabel(f"new {plot_comp}")
    plt.title(f"Old vs New: {plot_comp}")
    plt.tight_layout()
    out_plot = out_dir / f"scatter_{plot_comp}.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Plot: {out_plot}")

if __name__ == "__main__":
    main()
