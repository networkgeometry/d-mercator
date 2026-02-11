# Static Optimization Plan (No Algorithm Changes)

## Scope
- This plan maps the ranked bottlenecks in `docs/perf_static_report.md` to concrete implementation changes.
- All changes preserve existing algorithmic flow, stopping criteria, and floating-point types.
- All changes are implementation-level only (allocation/layout/lookups/micro-hoisting/loop structure).

## Bottleneck-to-Optimization Mapping

### B1/B2: `refine_angle` hot loops (`include/dmercator/refine/refinement.hpp:560-575`, `609-620`, `655-672`, `697-708`)
- Optimization A: Reuse per-call scratch buffers for neighbors/prefactors/proposals.
  - Category: **1. Allocation reduction**.
  - Preservation: same neighbor order and likelihood equations; only allocation site changes.
  - Cached/precomputed: per-instance scratch vectors reused across vertex refinements.
  - Expected gain: lower allocator overhead and less cache/memory churn.
- Optimization B: Move invariant computations outside candidate loops.
  - Category: **5. Loop restructuring**.
  - Preservation: uses the same formulas and evaluation order per candidate; only hoists candidate-invariant values.
  - Cached/precomputed: `pair_prefactor[v2]`, neighbor list, `inv_dim`, reusable temporary vectors.
  - Expected gain: reduced repeated work inside the hottest inner loops.

### B3: `compute_inferred_ensemble_expected_degrees` (`include/dmercator/inference/inference.hpp:340-378`)
- Optimization: Hoist loop-invariant scalar constants and avoid avoidable temporaries/reinitialization patterns.
  - Category: **4. Math micro-optimizations**, **5. Loop restructuring**.
  - Preservation: exact probability function remains unchanged.
  - Cached/precomputed: `inv_dim`, outer-loop references (`kappa1`, `pos1`), pre-sized result vector.
  - Expected gain: reduced constant factors in repeated `O(N^2)` passes.

### B4: Degree-class kappa inference (`include/dmercator/inference/inference.hpp:650-809`)
- Optimization: Build contiguous degree-class index arrays per iteration to reduce map lookups in nested loops.
  - Category: **2. Data layout**, **3. Lookup reduction**.
  - Preservation: same degree-class iteration order (ascending) and same updates.
  - Cached/precomputed: vector of degree classes, vector of class sizes, local references to kappa/expected-degree maps.
  - Expected gain: less pointer chasing and lower associative-container overhead.

### B5: Monte Carlo clustering (`include/dmercator/inference/inference.hpp:481-548`)
- Optimization: Hoist per-degree constants and avoid redundant map lookups inside MC loop.
  - Category: **4. Math micro-optimizations**, **3. Lookup reduction**.
  - Preservation: same random draws and same formulas; no approximations.
  - Cached/precomputed: class-specific kappa values and reused local scalars.
  - Expected gain: lower constant cost per Monte Carlo sample.

### B6/B7: Pairwise validation/simulation loops (`include/dmercator/embedding/orchestrator.hpp:286-339`, `include/dmercator/io/embedding_io.hpp:27-42`, `93-109`)
- Optimization A: Sequential neighbor-membership scan in pair loop instead of repeated `set::find`.
  - Category: **3. Lookup reduction**, **5. Loop restructuring**.
  - Preservation: identical pair iteration order and edge-membership outcome.
  - Cached/precomputed: sorted neighbor vector pointer per `v1` while iterating `v2` increasingly.
  - Expected gain: removes `O(log deg)` tree lookup from the innermost pair loop.
- Optimization B: Reuse preallocated containers for simulated adjacency setup.
  - Category: **1. Allocation reduction**.
  - Preservation: same sampled edges, same storage semantics.
  - Cached/precomputed: capacity reuse of simulated adjacency containers.
  - Expected gain: lower repeated allocation overhead in characterization mode.

### B8: Triangle/statistics passes (`include/dmercator/inference/inference.hpp:45-154`, `268-338`)
- Optimization: Reuse intersection buffers and avoid repetitive container growth churn.
  - Category: **1. Allocation reduction**.
  - Preservation: same set-intersection logic.
  - Cached/precomputed: reusable intersection vector capacity and temporary set capacities where possible.
  - Expected gain: reduced heap pressure in triangle-heavy loops.

### B9/B10: Initial ordering/load paths (`include/dmercator/refine/refinement.hpp:43-245`, `248-439`, `include/dmercator/init/initialization.hpp:202-294`)
- Optimization A: Reserve known-size temporary vectors and avoid repeated small-vector push-back churn.
  - Category: **1. Allocation reduction**.
  - Preservation: same values and ordering.
  - Cached/precomputed: pre-sized position vectors and neighbor buffers.
  - Expected gain: lower temporary allocation overhead.
- Optimization B: Keep contiguous adjacency iteration caches for frequent neighbor traversals.
  - Category: **2. Data layout**.
  - Preservation: adjacency remains semantically identical; set container remains authoritative for insertion/uniqueness.
  - Cached/precomputed: `vector<vector<int>>` neighbor lists built from sorted `set` adjacency.
  - Expected gain: improved locality and faster iteration in refinement/validation loops.

## Explicit Non-Changes
- No approximation of transcendental functions (`sin/cos/exp/log/pow`) is introduced.
- No change to convergence thresholds, optimization schedule, or random distribution definitions.
- No floating-point type changes.
- No new threading model introduced (OpenMP usage remains as currently present).

## Validation Mapping
- Every changed hot path will be compared via baseline-vs-optimized execution mode in the same binary.
- Validation checks:
  - C++ deterministic small-graph smoke comparison (kappa/r/theta or position components within tolerance).
  - Python end-to-end script (`python/validate_perf_preservation.py`) reporting RMSE and max absolute differences.

### Optimization-to-Check Traceability
- `refine_angle` allocation/buffer reuse (`include/dmercator/refine/refinement.hpp:560-575`, `655-672`): checked by C++ smoke (`tests/perf_preservation_smoke.cpp`) and Python baseline-vs-optimized comparison.
- Degree-class lookup reduction (`include/dmercator/inference/inference.hpp:650-809`): checked by C++ smoke and Python baseline-vs-optimized comparison.
- Pairwise adjacency lookup restructuring in connection-probability output (`include/dmercator/io/embedding_io.hpp:27-42`, `93-109`): checked by Python baseline-vs-optimized comparison on generated graphs.
- Adjacency flat-cache/data-layout changes (`include/dmercator/init/initialization.hpp:202-294`, `include/dmercator/refine/refinement.hpp:43-439`): checked by C++ smoke and Python baseline-vs-optimized comparison.
