#!/bin/bash
#SBATCH -J dmercator_topology
#SBATCH -c 8
#SBATCH -t 4000
#SBATCH -p mweber_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --open-mode=append
#SBATCH -o topology_%j.out
#SBATCH -e topology_%j.err

set -euo pipefail
set -x

JOB_DIR="${JOB_DIR:-${1:-}}"
if [[ -z "${JOB_DIR}" ]]; then
  echo "Usage: sbatch run_topology_validation_sbatch.sh [job_dir]" >&2
  exit 2
fi

module load gcc/12.2.0-fasrc01
module load cmake
module load cuda/12.9.1-fasrc01

module load python/3.10.12-fasrc01
source activate pt2.3.0_cuda12.1

nvidia-smi || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
GENERATOR_BUILD_DIR="${GENERATOR_BUILD_DIR:-${REPO_ROOT}/build_cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SPECTRAL_RANK="${SPECTRAL_RANK:-20}"
MIN_DEGREE_COUNT="${MIN_DEGREE_COUNT:-10}"
SIZE_LIMIT="${SIZE_LIMIT:-}"
FORCE="${FORCE:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"

if [[ ! -x "${GENERATOR_BUILD_DIR}/generate_sd" ]]; then
  cmake -S "${REPO_ROOT}" -B "${GENERATOR_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDMERCATOR_RELEASE_O3=ON \
    -DUSE_CUDA=OFF
  cmake --build "${GENERATOR_BUILD_DIR}" -j "${THREADS}"
fi

CMD=(
  python3
  "${REPO_ROOT}/python/topology_from_embeddings.py"
  --job-dir "${JOB_DIR}"
  --generator-binary "${GENERATOR_BUILD_DIR}/generate_sd"
  --spectral-rank "${SPECTRAL_RANK}"
  --min-degree-count "${MIN_DEGREE_COUNT}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${SIZE_LIMIT}" ]]; then
  CMD+=(--size-limit "${SIZE_LIMIT}")
fi

case "${FORCE}" in
  1|true|TRUE|yes|YES|y|Y)
    CMD+=(--force)
    ;;
  *)
    ;;
esac

"${CMD[@]}"
