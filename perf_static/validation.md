# Validation Plan and Harness

## Added harness

- New test: `tests/test_gpu_equivalence.cpp`
- Build wiring: `CMakeLists.txt` adds `test_gpu_equivalence` target when `BUILD_TESTING=ON` and `USE_CUDA=ON`.

## What the test compares

For `dim=1` and `dim=2`, using identical graph + seed + beta:

- CPU run (`CUDA_MODE=false`) vs GPU run (`CUDA_MODE=true`) on optimized path.
- Compared outputs:
  - `kappa` (max absolute difference with abs/rel tolerance)
  - coordinates:
    - `theta` for `dim=1` (circular absolute difference)
    - `d_positions` for `dim=2` (max absolute coordinate difference)
  - global scalars:
    - `beta` (from in-memory state)
    - `mu` (parsed from `.inf_coord` header)

## Tolerances / pass criteria

- Scalar (`beta`, `mu`): `abs_tol=1e-12`, `rel_tol=1e-12`
- Embedding vectors (`kappa`, coordinates): `abs_tol=1e-5`, `rel_tol=1e-5`

## Determinism and nondeterminism notes

- RNG semantics remain CPU-side (`std::mt19937`) for candidate generation and kappa updates.
- CUDA scoring/expected-degree summations preserve per-thread sequential accumulation order.
- Remaining small differences are floating-point backend differences (CPU vs GPU math library/instruction behavior), handled by the tolerances above.

## Objective/likelihood comparison

- The codebase does not currently expose a global final objective/likelihood scalar as a public output for direct CPU/GPU assertion.
- Harness therefore validates the final inferred state (`kappa`, coordinates) and key globals (`beta`, `mu`).

## Execution status for this change set

- Compilation and runtime execution were intentionally skipped per latest user instruction.
- The harness and build wiring are in place for deferred execution.

