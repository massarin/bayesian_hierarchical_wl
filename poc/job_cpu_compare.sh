#!/bin/bash
#SBATCH --job-name=bhwl_cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=poc/out/cpu_%j.out
#SBATCH --error=poc/out/cpu_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p poc/out

ENV=$HOME/envs/bhwl
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl-$USER
mkdir -p "$MPLCONFIGDIR"

echo "=== node $(hostname), $(nproc) cpus"
grep -m1 'model name' /proc/cpuinfo

# the same-node CPU counterpart to the GPU numbers: is the A40 buying anything?
for N in 100 1000; do
    echo; echo "=== N=$N, CPU ==="
    $ENV/bin/python poc/rung3_hierarchical.py --nlens $N --chains 8 --device cpu
done
