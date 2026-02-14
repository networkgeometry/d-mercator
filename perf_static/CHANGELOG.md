# Performance Refactor Changelog (Logical Commits)

## 1) GPU pair-prefactor preparation and refinement-state residency

- Files:
  - `include/dmercator/gpu/kernels.cuh`
  - `src/gpu/kernels.cu`
  - `include/dmercator/gpu/gpu_context.hpp`
  - `src/gpu/gpu_context.cu`
  - `include/dmercator/refine_positions.hpp`
- Why:
  - Static analysis flagged per-vertex host upload of `pair_prefactor` as avoidable overhead in refinement.
- What:
  - Added CUDA prep kernels to compute pair-prefactors on device from resident `kappa`.
  - Added `begin_refine_*` overloads that upload `kappa` once at refinement start.
  - Added `evaluate_refine_*_candidates_from_kappa(...)` GPU entry points.
  - Switched optimized refinement path to use these entry points.

## 2) Prepared expected-degree GPU path for kappa convergence loops

- Files:
  - `include/dmercator/gpu/gpu_context.hpp`
  - `src/gpu/gpu_context.cu`
  - `include/dmercator/infer_parameters.hpp`
- Why:
  - Static analysis flagged repeated upload of invariant `theta`/`positions` during `infer_kappas_given_beta_for_all_vertices`.
- What:
  - Added prepare/compute/clear expected-degree API:
    - `prepare_inferred_expected_degrees_*`
    - `compute_inferred_expected_degrees_*_prepared`
    - `clear_inferred_expected_degrees_state`
  - Wired kappa convergence loops to prepare once, upload only `kappa` per iteration, and fallback safely to CPU on failures.

## 3) Hot-path allocation reuse in optimized refinement

- Files:
  - `include/dmercator/engine.hpp`
  - `include/dmercator/refine_positions.hpp`
- Why:
  - Static analysis showed repeated vector allocation in inner candidate loops.
- What:
  - Added reusable scratch vectors for candidate angles/scores/flat positions.
  - Replaced per-call temporaries with reused buffers.
  - Made CPU pair-prefactor construction lazy (only for CPU scoring/fallback path).

## 4) CPU/GPU equivalence harness

- Files:
  - `tests/test_gpu_equivalence.cpp`
  - `CMakeLists.txt`
  - `perf_static/validation.md`
- Why:
  - Required correctness gate comparing CPU and GPU outputs on identical inputs/seeds.
- What:
  - Added test target (when `USE_CUDA=ON`) for dim=1 and dim=2 equivalence checks:
    - `kappa`, coordinates, `beta`, and `mu`.
  - Added tolerance policy and deterministic/nondeterministic notes.

## 5) Static-analysis deliverables

- Files:
  - `perf_static/report_hotspots.md`
  - `perf_static/plan_gpu.md`
- Why:
  - Required documentation of bottleneck discovery and implementation plan from static signals only.
- What:
  - Ranked hotspot report with complexity and GPU suitability.
  - Per-hotspot optimization plan with risk/mitigation notes.

