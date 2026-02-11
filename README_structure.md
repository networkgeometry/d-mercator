# Repository Structure (Embedding Code)

This refactor keeps the embedding behavior unchanged and reorganizes code by responsibility.

## Where to start

1. `apps/embed_sd.cpp` and `apps/generate_sd.cpp` (CLI entry points)
2. `include/dmercator/embedding/embedder.hpp` and `src/dmercator/embedding/embedder.cpp`
3. `include/dmercator/embedding/engine.hpp` (`embeddingSD_t` state + method declarations)
4. Module implementations included at the bottom of `engine.hpp`

## Folder responsibilities

- `include/dmercator/core/`
  - CLI generator adapter declaration and low-level engine helpers.
- `include/dmercator/embedding/`
  - Public embedding adapter + engine declaration + orchestration implementation.
- `include/dmercator/init/`
  - Initialization and graph/coordinate loading routines.
- `include/dmercator/inference/`
  - Parameter inference and random-ensemble/statistics routines.
- `include/dmercator/refine/`
  - Position initialization/refinement and pairwise likelihood routines.
- `include/dmercator/io/`
  - Coordinate CSV conversion and inferred-output serialization.
- `include/dmercator/geometry/`
  - Geometry helper utilities.
- `src/dmercator/`
  - `.cpp` implementations for thin CLI adapters.

## High-level embedding pipeline

`apps/embed_sd.cpp`  
-> `dmercator::embedding::Embedder::run_from_cli`  
-> parse CLI options (`parse_options`)  
-> `embeddingSD_t::embed(int dim)`  
-> `initialize()`  
-> `infer_parameters(dim)` (or `load_already_inferred_parameters(dim)` in refine mode)  
-> `infer_initial_positions(dim)`  
-> `refine_positions(dim)`  
-> `infer_kappas_given_beta_for_all_vertices(dim)`  
-> `save_inferred_coordinates(dim)` (+ optional validation outputs)  
-> `finalize()`
