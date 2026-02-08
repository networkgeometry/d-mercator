# Performance Static Report

## Scope
- Static analysis only of `include/embeddingSD.hpp` and directly used helpers (`include/integrate_expected_degree.hpp`, `include/readjust_positions.hpp`, build files).
- No program execution, benchmarks, tests, or profilers were run.

## Ranked Static Hotspots
1. `refine_positions()` / `refine_positions(int dim)` calling `refine_angle(...)`
   - Complexity:
     - 2D path: `O(N * A * (N + deg(v)))` pairwise likelihood terms, where `A ~= MIN_NB_ANGLES_TO_TRY * log N`.
     - D path: same structure, each pair also computes D-dimensional angle and multiple `pow/log`.
   - Why dominant:
     - Nested loops over vertices and candidate proposals.
     - Repeated expensive math (`pow`, `log`, `acos`) in inner loops.
     - D-dimensional variant currently recomputes pair-invariant denominator `pow(mu*kappa[v1]*kappa[v2], 1/d)` for every candidate.
   - Hoistable invariants:
     - Per-vertex-pair factors depending only on `(v1, v2, mu, kappa, dim, radius)`.
     - Neighbor index list for `v1` (avoid repeated tree iterator setup).
     - Reusable candidate buffer allocations.

2. `compute_inferred_ensemble_expected_degrees(int dim, double radius)` and `compute_inferred_ensemble_expected_degrees()`
   - Complexity: `O(N^2 * D)` (D path) and `O(N^2)` (2D path), repeated across kappa convergence iterations.
   - Why dominant:
     - Full all-pairs loop executed repeatedly inside `infer_kappas_given_beta_for_all_vertices(...)`.
     - Inner loop uses `pow` and geometric distance calculations.
   - Hoistable invariants:
     - `inv_dim = 1.0 / dim`.
     - Per-outer-loop constants (`kappa[v1]`, `theta[v1]`, `d_positions[v1]` references).

3. `generate_simulated_adjacency_list(int dim, bool random_positions)`
   - Complexity: `O(N^2 * D)` per generated graph.
   - Why dominant:
     - Full all-pairs Monte Carlo edge sampling.
     - Inner loop includes `pow` and angle computation.
   - Hoistable invariants:
     - `inv_dim = 1.0 / dim`.
     - Per-outer-loop references (`d_positions[v1]`, `kappa[v1]`).

4. `find_initial_ordering(std::vector<std::vector<double>>&, int dim)`
   - Complexity:
     - Sparse edge pass roughly `O(M * integral_cost)`.
     - Spectra solve superlinear in embedded core size.
     - Degree-1 reinsertion loop has repeated rotation matrix setup currently inside inner neighbor loop.
   - Why dominant:
     - Expensive Fortran-backed integrals and eigen solve.
     - Avoidable repeated matrix construction when attaching many degree-1 neighbors to same anchor vertex.
   - Hoistable invariants:
     - Per-anchor rotation matrix and unit-direction vector.
     - Fixed axis vector for rotations.

5. Degree-class parameter fitting:
   - `infer_kappas_given_beta_for_degree_class(...)`, `build_cumul_dist_for_mc_integration(...)`
   - Complexity: `O(I * C^2 * integral_cost)` where `C` is number of degree classes.
   - Why dominant:
     - Nested degree-class loops with repeated map lookups and integral/hypergeometric calls.
   - Hoistable invariants:
     - Cached map values per loop level.
     - Iteration-local temporaries to reduce repeated associative container lookups.

## Optimization Plan (Conservative First)
1. Cache pair-invariant factors inside `refine_angle` (2D + D)
   - Expected impact: High
   - Risk: Low
   - Regions: `include/embeddingSD.hpp` (`refine_angle(int)`, `refine_angle(int,int,double)`)
   - Semantics preservation:
     - Keep exact objective decomposition and formulas.
     - Only hoist terms independent of proposed angle and reuse them.

2. Remove repeated allocations/copies in hot loops
   - Expected impact: Medium
   - Risk: Low
   - Regions: `refine_angle(int,int,double)`, `find_initial_ordering(..., int dim)`, D-dimensional pairwise helper signature
   - Semantics preservation:
     - Replace pass-by-value with `const&` where values are read-only.
     - Reuse buffers with identical contents.

3. Hoist repeated rotation setup for degree-1 reinsertion
   - Expected impact: Medium
   - Risk: Low
   - Regions: `find_initial_ordering(std::vector<std::vector<double>>&, int dim)`
   - Semantics preservation:
     - Rotation matrix depends only on anchor vertex direction; moving construction outside inner neighbor loop is equivalent.

4. Tighten D-dimensional all-pairs loops with local invariants
   - Expected impact: Medium
   - Risk: Low
   - Regions: `compute_inferred_ensemble_expected_degrees(int, double)`, `generate_simulated_adjacency_list(int, bool)`
   - Semantics preservation:
     - Keep same formulas; only reuse outer-loop constants and references.

5. Build-system safe performance toggles (off by default)
   - Expected impact: Medium (opt-in)
   - Risk: Low
   - Regions: `CMakeLists.txt`, `python/CMakeLists.txt`
   - Semantics preservation:
     - Default behavior unchanged; optional `-march=native`/IPO/LTO and optional `-O3` gated behind CMake options.

## Remaining High-Risk Areas (Need Real Profiling Later)
- Whether `std::set<int>` adjacency storage is the main locality bottleneck versus math-heavy kernels.
- Relative time split between Spectra eigen solve vs refinement for different graph sizes/dimensions.
- Impact of degree-class map/set usage versus Fortran integral cost in parameter inference loops.
