# Embedding Duplication Audit (`d=1` vs `d>1`)

Scope: `embeddingSD_t` methods split across:
- `include/dmercator/inference/inference.hpp`
- `include/dmercator/refine/refinement.hpp`
- `include/dmercator/init/initialization.hpp`
- `include/dmercator/io/embedding_io.hpp`
- `include/dmercator/embedding/orchestrator.hpp`
- `include/dmercator/core/helpers.hpp`

| Function pair | Current location | Difference between `d=1` and `d>1` |
|---|---|---|
| `build_cumul_dist_for_mc_integration()` + `(int dim)` | `inference.hpp` | S1 closed form; higher dimensions use integral-based path |
| `compute_inferred_ensemble_expected_degrees()` + `(int dim, double radius)` | `inference.hpp` | S1 circular distance vs vector-angle distance |
| `compute_random_ensemble_clustering()` + `(int dim)` | `inference.hpp` | Sampling/distribution differs by geometry |
| `compute_random_ensemble_clustering_for_degree_class(int)` + `(int, int dim)` | `inference.hpp` | S1 hypergeometric form vs dimensional integral |
| `degree_of_random_vertex_and_prob_conn(int,double)` + `(int,double,int dim)` | `inference.hpp` | Probability normalization differs for `S^1` and `S^D` |
| `draw_random_angular_distance(int,int,double,double)` + `(int,int,double,double,int dim)` | `inference.hpp` | S1 circular draw vs dimensional draw |
| `generate_simulated_adjacency_list()` + `(int dim, bool)` | `orchestrator.hpp` | S1 uses `theta`; higher dimensions use `d_positions` |
| `infer_initial_positions()` + `(int dim)` | `refinement.hpp` | S1 EigenMap ordering + angular spacing; higher dimensions use vector initialization |
| `infer_kappas_given_beta_for_all_vertices()` + `(int dim)` | `inference.hpp` | Expected-degree update differs by geometry |
| `infer_kappas_given_beta_for_degree_class()` + `(int dim)` | `inference.hpp` | Connection probability and iteration schedule differ by geometry |
| `infer_parameters()` + `(int dim)` | `inference.hpp` | Beta search and clustering evaluation differ by geometry |
| `refine_positions()` + `(int dim)` | `refinement.hpp` | S1 angle refinement vs vector refinement |
| `load_already_inferred_parameters()` + `(int dim)` | `initialization.hpp` | S1 input schema (`theta`) vs higher-dimensional position vectors |
| `save_inferred_connection_probability()` + `(int dim)` | `embedding_io.hpp` | S1 circular distances vs vector/geodesic distances |
| `save_inferred_coordinates()` + `(int dim)` | `embedding_io.hpp` | Output schema differs (`theta` vs `pos_0..pos_D`) |

Notes:
- Canonical dim-aware methods are retained; no-argument overloads remain as compatibility wrappers.
- Specialized `dim == 1` branches remain only where file schema or geometry requires it.
