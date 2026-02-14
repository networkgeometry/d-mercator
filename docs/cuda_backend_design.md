# CUDA Backend Design (refactor_cpu)

## Summary
This branch now uses a persistent CUDA likelihood backend object that owns:
- CSR graph on device (`row_offsets`, `col_indices`)
- Device-resident model state (`kappa`, `theta`, `positions` in SoA)
- Reused candidate/result buffers for refinement and expected-degree passes

The embedding math is unchanged from CPU code paths. The refactor changes placement, transfer strategy, and kernel wiring only.

## Build and Run
- CPU-only build:
  - `cmake -S . -B build_cpu -DD_MERCATOR_CUDA=OFF`
- CUDA build:
  - `cmake -S . -B build_cuda -DD_MERCATOR_CUDA=ON`
  - Override architectures with `-DCMAKE_CUDA_ARCHITECTURES=...`

Runtime backend selection:
- Force CPU: `embed_sd ... -C`
- Request CUDA: `embed_sd ... -G`
- Deterministic CUDA mode on/off: `-D` / `-N`

Timing logs:
- Add `--timing_json` to print stage timings as one JSON line per run.

## Execution Model
1. `initialize()` builds CSR once and initializes a backend instance once.
2. Refinement uploads full state once per stage (`set_theta`/`set_positions_soa`, `set_kappa`).
3. Per-vertex refinement only transfers candidate buffers and (if accepted) a single updated vertex entry.
4. Kappa inference keeps CPU update logic (RNG and stopping criteria) intact and uses backend expected-degree kernels per iteration.

## Numerics and Tolerances
Because reductions are parallelized, floating-point summation order differs from CPU loops. Use these tolerances:
- S1 candidate score max abs diff: `1e-9`
- SD candidate score max abs diff: `1e-9`
- Expected-degree vector max abs diff: `1e-7`
- Final `kappa` max abs diff: `1e-6`
- Final S1 `theta` circular abs diff: `1e-6`
- Final SD coordinate max abs diff: `1e-6`

Use `python/benchmark_cpu_gpu.py` as the runtime harness for CPU/GPU comparisons and timings.

## Hot-Path Constraints Satisfied
- No `cudaMalloc/cudaFree` in per-vertex refinement loops.
- No O(N) host-device copies in per-vertex refinement loops.
- Graph and model state remain device-resident across embedding stages.
