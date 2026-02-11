# Validation Workflow (S^D generation + embedding comparison)

This repository now provides a reproducible validation pipeline that compares:
- ground truth coordinates from synthetic S^D generation,
- legacy embedding output,
- refactored embedding output.

## 1) Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Built executables used by the workflow:
- `build/generate_sd`
- `build/embed_sd`
- `build/mercator_legacy`
- `build/perf_preservation_smoke` (CTest target, when tests are enabled)

## 2) End-to-end validation (recommended)

Run d=1:

```bash
python3 python/validate_sd.py \
  --build-dir build \
  --output-dir validation_output/d1 \
  --dimension 1 \
  --nodes 300 \
  --seed 12345
```

Run d=2:

```bash
python3 python/validate_sd.py \
  --build-dir build \
  --output-dir validation_output/d2 \
  --dimension 2 \
  --nodes 220 \
  --seed 12345
```

The script performs:
1. hidden-variable generation (`hidden_variables.txt`),
2. synthetic network generation with ground truth via `generate_sd`,
3. embedding with `mercator_legacy`,
4. embedding with `embed_sd`,
5. alignment + metrics + assertions,
6. plotting.

Outputs (inside `--output-dir`):
- `coords_truth.csv`
- `coords_original.csv`
- `coords_refactored.csv`
- `metrics.json`
- `plots/`

## 3) Manual step-by-step commands

Example (`d=2`):

```bash
# hidden variables file must contain: kappa pos_0 ... pos_D when -t is used
build/generate_sd -d 2 -b 4.5 -s 12345 -o validation_manual/sd -t validation_manual/hidden_variables.txt

build/mercator_legacy -q -d 2 -s 12345 -b 4.5 -o validation_manual/original validation_manual/sd.edge

build/embed_sd -q -d 2 -s 12345 -b 4.5 -o validation_manual/refactored validation_manual/sd.edge

python3 python/validate_sd.py \
  --build-dir build \
  --output-dir validation_manual \
  --dimension 2 \
  --nodes 220 \
  --seed 12345
```

## 4) Quick integration tests

```bash
python3 -m unittest test/test_validation_pipeline.py
python3 -m unittest test/test_perf_preservation_pipeline.py
ctest --test-dir build -R perf_preservation_smoke --output-on-failure
```

This runs small d=1 and d=2 smoke validations.

## 5) Baseline vs Optimized Preservation Check (new)

The embedder now supports:
- `-M baseline`
- `-M optimized` (default)

Run a full preservation check (build + generate + embed twice + compare + plot):

```bash
python3 python/validate_perf_preservation.py \
  --build-dir build \
  --output-dir validation_output/perf_preservation \
  --dimension 2 \
  --nodes 80 \
  --seed 12345 \
  --beta 4.5
```

This script:
1. builds Release binaries,
2. generates a synthetic S^D network via `generate_sd`,
3. runs `embed_sd -M baseline`,
4. runs `embed_sd -M optimized`,
5. compares coordinate RMSE and max absolute error per field,
6. writes `perf_preservation_metrics.json` and a scatter plot.

To run only comparison (skip build):

```bash
python3 python/validate_perf_preservation.py \
  --build-dir build \
  --output-dir validation_output/perf_preservation \
  --skip-build
```

## 6) Direct old-vs-new output comparison

```bash
python3 python/compare_outputs.py \
  --old validation_output/quick/original.inf_coord \
  --new validation_output/quick/ref.inf_coord \
  --dimension 1 \
  --output-dir validation_output/quick/compare_d1
```

```bash
python3 python/compare_outputs.py \
  --old validation_output/quick/original2.inf_coord \
  --new validation_output/quick/ref2.inf_coord \
  --dimension 2 \
  --output-dir validation_output/quick/compare_d2 \
  --plot-component pos_0
```
