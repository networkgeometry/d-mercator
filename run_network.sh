#!/bin/bash
#SBATCH -J dmercator_network
#SBATCH -c 8 # Number of cores requested
#SBATCH -t 4000 # Runtime in minutes
#SBATCH -p mweber_gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=64000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o network_%j.out # Standard out goes to this file
#SBATCH -e network_%j.err # Standard err goes to this file

set -euo pipefail
set -x

if [[ ${#} -lt 2 ]]; then
  echo "Usage: sbatch run_network.sh [path_to_edgelist] [dimension] [is_validation_mode]" >&2
  exit 2
fi

EDGE_PATH="$1"
DIMENSION="$2"
VALIDATION_MODE="${3:-0}"

module load gcc/12.2.0-fasrc01
module load cmake
module load cuda/12.9.1-fasrc01

nvidia-smi || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
CUDA_BUILD_DIR="${CUDA_BUILD_DIR:-${REPO_ROOT}/build_cuda}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/benchmark_runs/${SLURM_JOB_ID}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"

if [[ ! -f "${EDGE_PATH}" ]]; then
  echo "Edgelist not found: ${EDGE_PATH}" >&2
  exit 1
fi

cmake -S "${REPO_ROOT}" -B "${CUDA_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDMERCATOR_RELEASE_O3=ON \
  -DUSE_CUDA=ON
cmake --build "${CUDA_BUILD_DIR}" -j "${THREADS}"

mkdir -p "${RESULTS_DIR}"

EDGE_BASENAME="$(basename "${EDGE_PATH}")"
EDGE_STEM="${EDGE_BASENAME%.*}"
OUT_ROOT="${RESULTS_DIR}/${EDGE_STEM}"

VALIDATION_FLAG=""
case "${VALIDATION_MODE}" in
  1|true|TRUE|yes|YES|y|Y)
    VALIDATION_FLAG="-v"
    ;;
  0|false|FALSE|no|NO|n|N|"")
    VALIDATION_FLAG=""
    ;;
  *)
    echo "Invalid is_validation_mode: ${VALIDATION_MODE} (use 0/1)" >&2
    exit 2
    ;;
 esac

"${CUDA_BUILD_DIR}/embed_sd" \
  -q \
  -d "${DIMENSION}" \
  -G -D \
  ${VALIDATION_FLAG} \
  -o "${OUT_ROOT}" \
  "${EDGE_PATH}"
