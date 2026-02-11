# Embedding Benchmark Workflow

This benchmark suite measures **baseline vs optimized** embedding runtime with stage-level timings,
and enforces coordinate-equivalence checks.

## Build (Release, O3)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDMERCATOR_RELEASE_O3=ON
cmake --build build -j
```

## Python dependencies

`python/run_benchmarks.py` and `python/compare_coordinates.py` require:
- `numpy`
- `pandas`
- `matplotlib` (only for plots; optional with `--no-plots`)

## One-command benchmark run

```bash
python3 python/run_benchmarks.py --out bench --reps 5
```

Defaults:
- `N`: `500,2000,8000`
- `d`: `1,2`
- fixed deterministic seeds
- strict equivalence tolerances (`rmse_tol=1e-8`, `max_abs_tol=1e-6`)

## Quick smoke run

```bash
python3 python/run_benchmarks.py --out bench_smoke --reps 1 --sizes 300,800 --dimensions 1,2
```

If plotting is unavailable in your environment:

```bash
python3 python/run_benchmarks.py --out bench_smoke --reps 1 --sizes 300,800 --dimensions 1,2 --no-plots
```

## Output layout

Under `--out` (default `bench`):
- `results.csv`: median/IQR timings and speedups per `(N,d)`
- `results.json`: full benchmark payload
- `timing_raw.json`: per-run timing records
- `equivalence_results.json`: alignment + error metrics per `(N,d)`
- `plots/speedup_d*.png`: speedup vs N
- `plots/stage_breakdown_d*.png`: stacked stage-time bars
- `artifacts/`: generated graphs, per-run coordinate files, debug info on failures

## Manual coordinate comparison

```bash
python3 python/compare_coordinates.py \
  --baseline bench/artifacts/d2_n500_seed*/embed_baseline_rep00.coords.csv \
  --optimized bench/artifacts/d2_n500_seed*/embed_optimized_rep00.coords.csv \
  --dimension 2 \
  --rmse-tol 1e-8 \
  --max-abs-tol 1e-6 \
  --json-out bench/compare_example.json
```

## Timing JSON directly from embedder

The embedder supports:
- `--mode baseline|optimized`
- `--timing_json`

Example:

```bash
build/embed_sd -q -d 2 -s 12345 -b 4.5 --mode optimized --timing_json -o /tmp/run /path/to/graph.edge
```

This prints one JSON line with:
- `total_time_ms`
- `initialization_ms`
- `parameter_inference_ms`
- `initial_positions_ms`
- `refining_positions_ms`
- `adjusting_kappas_ms`
- `io_ms`
