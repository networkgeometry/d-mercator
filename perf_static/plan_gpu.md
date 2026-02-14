# GPU Optimization Plan (Static Analysis Driven)

No hotspot selection in this plan used profiler/benchmark discovery.

## Hotspot A: Refinement candidate scoring

- Target:
  - `refine_angle(int)` and `refine_angle(int dim, int, double)` in `include/dmercator/refine_positions.hpp`
  - GPU dataflow in `src/gpu/gpu_context.cu`
- Chosen strategy:
  - `(A)` extend CUDA kernels
  - `(B)` reduce per-vertex transfer overhead
  - `(E)` keep invariant state resident on GPU
- New/used kernel signatures:
  - `prepare_pair_prefactor_s1_kernel(const double *kappa, int nb_vertices, int v1, double prefactor_over_kappa_v1, double *out_pair_prefactor)`
  - `prepare_pair_prefactor_sd_kernel(const double *kappa, int nb_vertices, int v1, double mu, double radius, double inv_dim, double *out_pair_prefactor)`
  - existing candidate-scoring kernels unchanged for math semantics.
- Data layout:
  - Keep `theta` / flattened `positions` and `kappa` resident in device buffers during refinement phase.
  - Continue candidate arrays as contiguous flat host vectors copied once per vertex.
- Gain mechanism:
  - Remove full `O(N)` pair-prefactor host upload per vertex.
  - Replace with on-device pair-prefactor preparation kernel.
  - Reuse host scratch vectors for candidate buffers.
- Correctness risks + mitigation:
  - Risk: floating-point expression reassociation.
  - Mitigation: preserved original formulas; only moved equivalent pair-prefactor computation location (CPU→GPU prep kernel).
  - Determinism: update order of vertices/candidates unchanged; RNG remains CPU-side.

## Hotspot B: Repeated expected-degree passes during kappa iterations

- Target:
  - `infer_kappas_given_beta_for_all_vertices(int dim)` in `include/dmercator/infer_parameters.hpp`
  - expected-degree GPU helpers in `src/gpu/gpu_context.cu`
- Chosen strategy:
  - `(E)` keep static state resident on GPU through convergence loop
  - `(A)` extend GPU API for prepared-state compute
- New GPU API flow:
  - Prepare once:
    - `prepare_inferred_expected_degrees_s1(theta)` or
    - `prepare_inferred_expected_degrees_sd(dim, positions)`
  - Per iteration:
    - upload only `kappa`
    - call `compute_inferred_expected_degrees_*_prepared(...)`
  - Cleanup:
    - `clear_inferred_expected_degrees_state()`
- Gain mechanism:
  - Avoid repeated upload/flatten of invariant geometry every iteration.
  - Preserve same per-iteration kappa update logic on CPU.
- Correctness risks + mitigation:
  - Risk: stale prepared state if dimension/size mismatch.
  - Mitigation: explicit size/dim checks and fallback to CPU on any failure.
  - Determinism: same loop order and stopping criteria; only transport/caching changed.

## Hotspot C: Hot-path allocations in refinement loops

- Target:
  - `refine_angle` optimized branches in `include/dmercator/refine_positions.hpp`
  - class scratch storage in `include/dmercator/engine.hpp`
- Chosen strategy:
  - `(B)` batching/reuse at host side
- Changes:
  - Reuse scratch vectors for:
    - candidate angles
    - candidate scores
    - flattened candidate positions
  - Lazy-build pair-prefactor on CPU only when GPU scoring is unavailable/fallback path taken.
- Gain mechanism:
  - Eliminate repeated allocations/resizes in inner refinement path.
- Correctness risks + mitigation:
  - Risk: stale scratch content.
  - Mitigation: explicit `clear()/assign()/fill()` before reuse, preserving computed values and order.

## Deferred (not implemented to preserve semantics/data structures)

- GPU random graph generation (`generate_simulated_adjacency_list`) was not moved:
  - would require RNG stream changes that risk violating seed semantics.
- Set-based triangle/statistics analysis was not structurally rewritten:
  - requires algorithm/data-structure changes beyond logic-preserving scope.

