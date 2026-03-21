#!/bin/bash
#SBATCH -J parallel_dmercator_gpu
#SBATCH -c 8 # Number of cores requested
#SBATCH -t 4000 # Runtime in minutes
#SBATCH -p mweber_gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=64000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o test_gpu_new_%j.out # Standard out goes to this file
#SBATCH -e test_gpu_new_%j.err # Standard err goes to this file

set -euo pipefail
set -x

module load gcc/12.2.0-fasrc01
module load cmake
module load cuda/12.9.1-fasrc01

module load python/3.10.12-fasrc01
source activate pt2.3.0_cuda12.1

nvidia-smi || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

CUDA_BUILD_DIR="${CUDA_BUILD_DIR:-${REPO_ROOT}/build_cuda}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/benchmark_runs/${SLURM_JOB_ID}}"

SIZES="${SIZES:-50000,100000,500000,1000000}"
DIMENSION="${DIMENSION:-1}"
SAMPLE_COUNTS="${SAMPLE_COUNTS:-16,64,256}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-2.5}"
MEAN_DEGREE="${MEAN_DEGREE:-10.0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"

cmake -S "${REPO_ROOT}" -B "${CUDA_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDMERCATOR_RELEASE_O3=ON \
  -DUSE_CUDA=ON
cmake --build "${CUDA_BUILD_DIR}" -j "${THREADS}"

mkdir -p "${RESULTS_DIR}"

python3 "${REPO_ROOT}/python/benchmark_negative_sampling_cpu_gpu.py" \
  --backends gpu \
  --generator-build-dir "${CUDA_BUILD_DIR}" \
  --gpu-build-dir "${CUDA_BUILD_DIR}" \
  --out-dir "${RESULTS_DIR}" \
  --sizes "${SIZES}" \
  --dimensions "${DIMENSION}" \
  --sample-counts "${SAMPLE_COUNTS}" \
  --seed "${SEED}" \
  --beta-mode two-times-dim \
  --gamma "${GAMMA}" \
  --mean-degree "${MEAN_DEGREE}" \
  --quality-size-limit 10000 \
  --omp-threads "${OMP_NUM_THREADS}"
