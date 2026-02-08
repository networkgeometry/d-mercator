# CODEX Work Log

## What Robert likes
- Conservative, semantics-preserving performance changes.
- Static analysis only (no runtime benchmarking/profiling/execution).
- Minimal, reviewable diffs with clear rationale.
- Avoid invasive API changes unless compatibility is preserved.
- IEEE-safe behavior (no fast-math shortcuts).
- Explicit hotspot ranking with Big-O and exact code regions.

## Potential mistakes / risks
- Static-only optimization can mis-rank true runtime hotspots without traces.
- Refactoring inner floating-point loops can change operation order slightly.
- Caching in hot loops must not accidentally use stale values after updates.
- Hidden assumptions around adjacency representation (`std::set`) may affect ordering/edge-case behavior.
- D-dimensional helper functions currently pass vectors by value; changing signatures must remain source-compatible.
- Build flag changes must stay portable and default-safe.
- D-dimensional cached pair prefactors now store `radius / pow(mu*kappa[v1]*kappa[v2], 1/d)` once per vertex-update; this is mathematically identical but can still introduce tiny round-off differences relative to recomputing each use.

## Decisions & rationale
- Focus first on `embed` call-chain hotspots: `refine_positions`/`refine_angle`, `compute_inferred_ensemble_expected_degrees`, and `generate_simulated_adjacency_list`.
- Prioritize exact invariant hoisting and redundant work removal over algorithmic rewrites.
- Keep data structures unchanged in this pass (`std::set` adjacency), unless a change is clearly low-risk and local.
- Created `PERF_STATIC_REPORT.md` with ranked hotspots and a conservative, impact/risk-scored optimization plan before code edits.
- Refactored `refine_angle(int)` to cache per-`v1` pair prefactors and reuse a local likelihood lambda, preserving the same two-term likelihood decomposition (all-pairs non-neighbor term + neighbor correction).
- Refactored `refine_angle(int dim, int v1, double radius)` to cache pair prefactors and reuse one proposal buffer, and changed D pairwise helper args to `const&` to eliminate vector copies.
- Hoisted D-dimensional degree-1 reinsertion rotation setup in `find_initial_ordering(..., int dim)` so rotation matrix construction is once per anchor vertex instead of once per attached degree-1 node.
- Hoisted loop invariants in D all-pairs kernels (`compute_inferred_ensemble_expected_degrees` and `generate_simulated_adjacency_list`) and removed per-iteration position copies via `const&`.
