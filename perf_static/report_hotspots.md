# Static Hotspot Report (No Profiling Used)

This report ranks likely bottlenecks from static code inspection only.

## 1) Refinement candidate scoring (`include/dmercator/refine_positions.hpp`)

- Functions:
  - `embeddingSD_t::refine_angle(int v1)` (S1)
  - `embeddingSD_t::refine_angle(int dim, int v1, double radius)` (S^D)
- Dominant loops:
  - Per vertex `v1`: evaluate `O(C)` candidate angles/positions.
  - Per candidate: full `O(N)` pass over all vertices plus `O(deg(v1))` neighbor correction.
  - Net static complexity: `O(N * C * (N + deg))`, with `C ~ MIN_NB_ANGLES_TO_TRY * log(N)`.
- Hot math in inner loops:
  - `pow`, `log`, `acos`, `sqrt`, trigonometric ops.
- Runtime-dominant structures:
  - `theta`, `d_positions`, `kappa`, adjacency neighbors.
- GPU suitability:
  - Compute-heavy, embarrassingly parallel across candidates.
  - Reduction-light (one score per candidate).
  - Main risk is transfer overhead if candidate context is copied each vertex.

## 2) Expected-degree recomputation inside kappa updates (`include/dmercator/infer_parameters.hpp` + `include/dmercator/validation.hpp`)

- Functions:
  - `embeddingSD_t::infer_kappas_given_beta_for_all_vertices(int dim)`
  - `embeddingSD_t::compute_inferred_ensemble_expected_degrees(...)`
- Dominant loops:
  - Repeated convergence iterations (`KAPPA_MAX_NB_ITER_CONV` bound).
  - Each iteration performs all-pairs expected-degree pass:
    - S1: `O(N^2)` with `pow` per pair.
    - S^D: `O(N^2 * D)` with angular-distance math (`acos/sqrt`) + `pow`.
- Runtime-dominant structures:
  - `theta`/`d_positions`, `kappa`, `inferred_ensemble_expected_degree`.
- GPU suitability:
  - Highly parallel all-pairs per `v1`.
  - Compute-heavy with predictable access pattern.
  - Strong benefit from keeping static geometry on device across iterations.

## 3) Ensemble generation/validation passes (`include/dmercator/validation.hpp`)

- Functions:
  - `generate_simulated_adjacency_list(int dim, bool random_positions)`
  - `analyze_simulated_adjacency_list()`
  - `save_inferred_connection_probability(int dim)` (all-pairs pass)
- Dominant loops:
  - Repeated all-pairs probabilities (`O(N^2)` / `O(N^2 * D)`).
  - Triangle/statistics passes over adjacency sets (intersection-heavy).
  - Characterization loop repeats graph generation many times.
- Runtime-dominant structures:
  - `std::set<int>` adjacency containers, `simulated_*` vectors/maps.
- GPU suitability:
  - Pairwise probability computation is GPU-suitable.
  - Graph sampling step is RNG-semantics-sensitive (current CPU RNG semantics should be preserved).
  - Set-based triangle analysis is branchy/scatter-heavy and less GPU-friendly without logic/data-structure redesign.

## Additional static signals observed

- Repeated host↔device transfers in existing GPU path (`src/gpu/gpu_context.cu`), especially:
  - Per-vertex pair-prefactor upload in refinement.
  - Re-upload of static geometry arrays in repeated expected-degree iterations.
- Repeated per-call allocations in refinement candidate buffers (hot path).

