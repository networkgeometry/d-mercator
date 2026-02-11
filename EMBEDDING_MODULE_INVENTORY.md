# Embedding Module Inventory

## Entry points
- `src/embeddingSD_unix.cpp`: legacy/default mercator CLI entry point.
- `apps/embed_sd.cpp`: refactored embedder CLI entry point.
- `apps/mercator_legacy.cpp`: baseline legacy embedder executable.
- `src/generatingSD_unix.cpp`: legacy/default generator CLI entry point.
- `apps/generate_sd.cpp`: refactored generator CLI entry point.

## Core embedding modules
- `include/dmercator/embedding/engine.hpp`: `embeddingSD_t` declarations and composed implementation includes.
- `include/dmercator/embedding/orchestrator.hpp`: top-level orchestration (`embed(int dim)`), runtime reporting, and simulated adjacency generation.
- `include/dmercator/init/initialization.hpp`: initialization, graph loading, prior-coordinate loading, and vertex ordering.
- `include/dmercator/inference/inference.hpp`: beta/kappa parameter inference and random-ensemble statistical support routines.
- `include/dmercator/refine/refinement.hpp`: pairwise likelihood, initial position inference, and coordinate refinement loops.
- `include/dmercator/io/embedding_io.hpp`: inferred output serialization and validation output files.
- `include/dmercator/core/helpers.hpp`: connectivity checks, geometry/math helpers, and kappa export.

## Public wrappers
- `include/embeddingSD.hpp`: backward-compatible include wrapper.
- `include/dmercator/embedding/embedder.hpp`: refactored embedding CLI adapter.
- `include/dmercator/core/generator.hpp`: refactored generator CLI adapter.

## d=1 vs d>1 structural forks
- Canonical dim-aware methods are used, with `dim==1` branches in places where file schema or geometry formulas differ.
- Backward-compatible no-arg wrappers remain for legacy call sites.
