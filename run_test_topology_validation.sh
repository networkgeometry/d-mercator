#!/bin/bash
#SBATCH -J dmercator_topology_test
#SBATCH -c 4
#SBATCH -t 30
#SBATCH -p mweber_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --open-mode=append
#SBATCH -o test_topology_%j.out
#SBATCH -e test_topology_%j.err

set -euo pipefail
set -x

module load gcc/12.2.0-fasrc01
module load cmake
module load cuda/12.9.1-fasrc01

module load python/3.10.12-fasrc01
source activate pt2.3.0_cuda12.1

nvidia-smi || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

cd "${REPO_ROOT}"
python3 -u -m unittest -v test.test_topology_from_embeddings
