# Mistakes Log

## 2026-02-11
- Assumed `test/generate_and_benchmark_s1_cpu_gpu.py` existed because it was visible in IDE context, but it is not present in this checkout (`sed` failed with "No such file or directory"). I switched to auditing the actual files listed by `rg --files`.
- Kept unconditional `-fopenmp` in Release flags initially; this broke local Clang builds. I changed CMake to use `find_package(OpenMP COMPONENTS CXX)` and link OpenMP only when available.
- Wrote an initial shell smoke command that incorrectly passed loop variables into a heredoc Python snippet (`command not found` noise). I replaced it with an `awk`-based angle generator and re-ran the smoke test.
- Replaced `load_already_inferred_parameters()` and `save_inferred_coordinates()` with wrappers too early, before preserving the S1-specific file schema inside the `int dim` versions. This would have broken `dim=1` parsing/output. I fixed it by adding explicit `if (dim == 1)` branches in the canonical functions, then re-ran build and smoke parity checks.
- Moved CLI adapter implementations to `.cpp` while still including legacy header-defined implementations from both adapter `.cpp` and app translation units, which caused duplicate linker symbols. I fixed it by making `include/dmercator/{embedding/embedder.hpp,core/generator.hpp}` declaration-only and moving legacy includes into the adapter `.cpp` files.
