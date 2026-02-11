# Static Performance Report (CPU Pipeline)

## Scope and Method
- Static analysis only.
- No execution, benchmarking, profiling, or timing measurements were performed.
- Analysis targets the current embedding pipeline in `embed_sd`/`mercator` code paths.

## 1) Entry Points and Pipeline Map

### Main embedding entry points
- `apps/embed_sd.cpp:3-6` -> `dmercator::embedding::Embedder::run_from_cli`.
- `src/dmercator/embedding/embedder.cpp:12-33` creates `embeddingSD_t`, parses CLI, runs `embed()`, exports CSV.
- Legacy-compatible entry also exists: `src/embeddingSD_unix.cpp:35-41`.
- CLI parsing for embedding options: `include/embeddingSD_unix.hpp:68-140`.

### Pipeline stages (from `embeddingSD_t::embed`)
- Stage A: Initialization (`include/dmercator/embedding/orchestrator.hpp:23`, `include/dmercator/init/initialization.hpp`).
- Stage B: Parameter inference (`include/dmercator/embedding/orchestrator.hpp:53`, `include/dmercator/inference/inference.hpp:816-1031`).
- Stage C: Initial position inference (`include/dmercator/embedding/orchestrator.hpp:60`, `include/dmercator/refine/refinement.hpp:442-539`).
- Stage D: Position refinement (`include/dmercator/embedding/orchestrator.hpp:66`, `include/dmercator/refine/refinement.hpp:722-783`).
- Stage E: Kappa post-adjustment (`include/dmercator/embedding/orchestrator.hpp:72`, `include/dmercator/inference/inference.hpp:556-648`).
- Stage F: Output/validation/characterization (`include/dmercator/embedding/orchestrator.hpp:81-95`, `include/dmercator/io/embedding_io.hpp`).

## 2) Call Graph and Hot-Loop Candidates

### Stage A: Initialization
- `load_edgelist` (`include/dmercator/init/initialization.hpp:202-294`)
  - Loop depth: 1 over input lines; inner map/set operations.
  - Bounds: `E` file rows, dynamic `N` growth.
  - Data structures: `std::map<string,int>`, `std::vector<std::set<int>>`.
  - Expensive ops: repeated map lookups/inserts, set inserts.
- `extract_onion_decomposition` (`include/dmercator/embedding/orchestrator.hpp:101-170`)
  - Loop depth: 3 (`while DegreeSet` -> `while layer` -> neighbors).
  - Bounds: roughly `N + E` with many ordered-set operations.
  - Data structures: `std::set<pair<int,int>>`, adjacency sets.

### Stage B: Parameter inference
- `infer_kappas_given_beta_for_degree_class` (`include/dmercator/inference/inference.hpp:650-809`)
  - Loop depth: 3 (`iteration` x `degree classes i` x `degree classes j`).
  - Bounds: `I_kappa * C^2` where `C` is number of degree classes.
  - Data structures: `std::set<int>`, `std::map<int,double>`, `std::map<int,vector<int>>`.
  - Expensive ops: repeated map lookups, `hyp2f1a`, `compute_integral_expected_degree_dimensions`, `pow`.
- `build_cumul_dist_for_mc_integration` (`include/dmercator/inference/inference.hpp:156-261`)
  - Loop depth: 2 over degree classes, repeated per beta iteration.
  - Bounds: `C^2`.
  - Data structures: nested maps (`cumul_prob_kgkp` is `map<int,map<double,int>>`).
- `compute_random_ensemble_clustering_for_degree_class` (`include/dmercator/inference/inference.hpp:481-548`)
  - Loop depth: Monte Carlo loop (`EXP_CLUST_NB_INTEGRATION_MC_STEPS`) plus binary searches in angle sampling.
  - Bounds: `MC * log(PI/eps)`.
  - Data structures: map lookups for degree-class kappas and cumulative distributions.
  - Expensive ops: RNG, `pow`, `hyp2f1a` / integral calls, angle generation.

### Stage C: Initial positions
- `find_initial_ordering` 1D and D (`include/dmercator/refine/refinement.hpp:43-439`)
  - Loop depth: edge loops + eigen solve setup + degree-one reinsertion loops.
  - Bounds: edge pass `O(E)` plus spectral solve on `N_{k>1}`.
  - Data structures: sparse matrix, adjacency sets, temporary vectors.
  - Expensive ops: repeated integral calls per edge, `sin`, `exp`, eigen decomposition.
- `infer_initial_positions` 1D (`include/dmercator/refine/refinement.hpp:442-533`)
  - Loop depth: over ordered vertices with numerical integration loop over `EXP_DIST_NB_INTEGRATION_STEPS`.
  - Bounds: `N * S` where `S=EXP_DIST_NB_INTEGRATION_STEPS`.
  - Expensive ops: `exp`, `pow`, membership lookup `adjacency_list[v0].find(v1)` in inner path.

### Stage D: Refinement
- `refine_angle(int v1)` (`include/dmercator/refine/refinement.hpp:542-632`)
  - Loop depth: `candidate angles` x `all vertices` (+ neighbor pass).
  - Bounds: `A * (N + deg(v1))`, `A ~= MIN_NB_ANGLES_TO_TRY * log N`.
  - Data structures: per-call temporary `neighbors` + `pair_prefactor` vectors.
  - Expensive ops: `pow`, `log`, `sin`, `cos`, repeated allocation.
- `refine_angle(int dim, int v1, double radius)` (`include/dmercator/refine/refinement.hpp:634-720`)
  - Loop depth: `candidate positions` x `all vertices` (+ neighbor pass) x `D` for angle calc.
  - Bounds: `A * (N + deg(v1)) * D`.
  - Data structures: per-call temporary vectors (`neighbors`, `pair_prefactor`, `mean_vector`, `proposed_position`).
  - Expensive ops: `compute_angle_d_vectors` (dot + norms + `acos`), `pow`, `log`.

### Stage E/F: Post-adjustment and validation outputs
- `compute_inferred_ensemble_expected_degrees` (`include/dmercator/inference/inference.hpp:340-378`)
  - Loop depth: nested all-pairs.
  - Bounds: `N*(N-1)/2` per iteration; called repeatedly in kappa convergence loops.
  - Data structures: vectors.
  - Expensive ops: `pow`, angle computation (`compute_angle_d_vectors`) in D.
- `save_inferred_connection_probability` (`include/dmercator/io/embedding_io.hpp:4-137`)
  - Loop depth: nested all-pairs.
  - Bounds: `N*(N-1)/2`.
  - Data structures: map bins + adjacency set membership checks.
  - Expensive ops: `pow`, repeated `adjacency_list[v1].find(v2)`.
- `generate_simulated_adjacency_list` (`include/dmercator/embedding/orchestrator.hpp:286-345`)
  - Loop depth: nested all-pairs, called many times in characterization.
  - Bounds: `N*(N-1)/2` per generated graph.
  - Data structures: simulated adjacency as `vector<set<int>>`.
  - Expensive ops: angle + `pow` + random sampling + set inserts.
- `compute_clustering` and `analyze_simulated_adjacency_list` (`include/dmercator/inference/inference.hpp:45-154`, `268-338`)
  - Loop depth: triangle-like loops with repeated `set_intersection` and temp containers.
  - Bounds: approximately sum over vertices of neighbor-neighbor operations.
  - Data structures: set-based adjacency, temporary `neighbors_v2` set and `intersection` vector.

## 3) Complexity and Memory Behavior (Static Reasoning)

### Candidate summary
- `refine_angle` (1D and D):
  - Time: `O(N * A * (N + avg_deg))`, D-path multiplies by angle-cost `O(D)`.
  - Constant-factor killers: `pow/log/acos`, repeated vector allocations, full `N` scan per candidate.
  - Memory behavior: repeated heap churn (`neighbors`, `pair_prefactor`, temp vectors) and pointer-heavy adjacency traversal.

- `compute_inferred_ensemble_expected_degrees`:
  - Time: `O(I_post * N^2)` where `I_post` is kappa-update iteration count.
  - Constant-factor killers: repeated transcendental math in inner pair loop.
  - Memory behavior: contiguous vectors (good locality) but repeated full-array zeroing and compute.

- `infer_kappas_given_beta_for_degree_class`:
  - Time: `O(I_kappa * C^2 * integral_cost)`.
  - Constant-factor killers: repeated `std::map`/`std::set` lookups inside nested loops.
  - Memory behavior: pointer-chasing ordered containers hurt cache locality.

- `compute_random_ensemble_clustering_for_degree_class`:
  - Time: `O(MC * (sampling + integral_cost))` per degree class.
  - Constant-factor killers: repeated RNG, binary search loops in angular sampling, expensive special functions.
  - Synchronization: OpenMP reductions at `include/dmercator/inference/inference.hpp:493` and `525`.

- `save_inferred_connection_probability` and `generate_simulated_adjacency_list`:
  - Time: `O(N^2)` each.
  - Constant-factor killers: `pow` in inner loops; repeated set membership checks (`find`) in output validation path.
  - Memory behavior: set-based adjacency (`std::set`) causes pointer chasing and branch-heavy traversal.

- `compute_clustering` / `analyze_simulated_adjacency_list`:
  - Time: near triangle-count style with repeated set intersections.
  - Constant-factor killers: constructing temporary sets/vectors repeatedly (`neighbors_v2`, `intersection`).
  - Memory behavior: repeated allocations and low locality.

- `load_edgelist`:
  - Time: `O(E * log N + E * log deg)` due map/set operations.
  - Constant-factor killers: string map lookups per endpoint and set insertions.
  - Memory behavior: fragmented allocations in maps/sets.

### Synchronization/parallel notes
- OpenMP is present in clustering integration loops (`inference.hpp:493`, `525`), with reduction synchronization.
- Shared RNG engine (`engine`) is accessed from those parallel regions via helper calls, which is a potential contention/correctness risk and can also limit scalability.

## 4) Prioritized Bottleneck List (Top 10)

1. `include/dmercator/refine/refinement.hpp:655-672`, `697-708` (D-dimensional local likelihood in `refine_angle`)  
   Why: deepest nested compute path (`A * N`) with `compute_angle_d_vectors` + `pow` + `log` in inner loop.  
   Fix category: Allocation reduction, math micro-optimizations, loop restructuring.  
   Numerical risk: Medium.

2. `include/dmercator/refine/refinement.hpp:560-575`, `609-620` (1D local likelihood in `refine_angle`)  
   Why: same refinement structure in 1D, repeated full-graph scans and transcendentals per proposal.  
   Fix category: Allocation reduction, loop restructuring.  
   Numerical risk: Low/Medium.

3. `include/dmercator/inference/inference.hpp:340-378` (`compute_inferred_ensemble_expected_degrees`)  
   Why: repeated `O(N^2)` all-pairs inside post-inference kappa loop.  
   Fix category: Math micro-optimizations, loop restructuring.  
   Numerical risk: Low/Medium.

4. `include/dmercator/inference/inference.hpp:650-809` (`infer_kappas_given_beta_for_degree_class`)  
   Why: `O(I_kappa * C^2)` with expensive map lookups and integral/hypergeometric calls.  
   Fix category: Lookup reduction, data layout (degree-class indexing vectors).  
   Numerical risk: Low.

5. `include/dmercator/inference/inference.hpp:481-548` (`compute_random_ensemble_clustering_for_degree_class`)  
   Why: Monte Carlo loop + repeated angle sampling + expensive special functions.  
   Fix category: Math micro-optimizations, loop restructuring.  
   Numerical risk: Medium.

6. `include/dmercator/embedding/orchestrator.hpp:286-339` (`generate_simulated_adjacency_list`)  
   Why: all-pairs `O(N^2)` repeated per generated graph in characterization mode.  
   Fix category: Math micro-optimizations, allocation reduction.  
   Numerical risk: Low/Medium.

7. `include/dmercator/io/embedding_io.hpp:27-42`, `93-109` (`save_inferred_connection_probability`)  
   Why: all-pairs loop with repeated `adjacency_list[v1].find(v2)` tree lookups in inner loop.  
   Fix category: Lookup reduction, loop restructuring.  
   Numerical risk: Low.

8. `include/dmercator/inference/inference.hpp:268-338` and `45-154` (`compute_clustering`, `analyze_simulated_adjacency_list`)  
   Why: repeated temporary set/vector creation and set intersections in high-iteration loops.  
   Fix category: Allocation reduction, data layout.  
   Numerical risk: Low.

9. `include/dmercator/refine/refinement.hpp:43-245` and `248-439` (`find_initial_ordering`)  
   Why: integral-heavy edge loop + sparse eigensolver setup + repeated temporary vector construction.  
   Fix category: Allocation reduction, loop restructuring.  
   Numerical risk: Low/Medium.

10. `include/dmercator/init/initialization.hpp:202-294` (`load_edgelist`)  
    Why: map/set insertion and string-key lookups on every input edge endpoint.  
    Fix category: Data layout (contiguous ids), allocation reduction.  
    Numerical risk: Low.
