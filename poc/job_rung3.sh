#!/bin/bash
#SBATCH --job-name=bhwl_rung3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=poc/out/rung3_%j.out
#SBATCH --error=poc/out/rung3_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p poc/out

ENV=$HOME/envs/bhwl
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl-$USER
mkdir -p "$MPLCONFIGDIR"

echo "=== node $(hostname)  job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

$ENV/bin/python - <<'PY'
import jax
print("jax", jax.__version__, "| devices:", jax.devices())
PY

echo; echo "=== pytest (rung 0/1) on GPU ==="
$ENV/bin/python -m pytest poc/test_physics.py -q 2>&1 | tail -3

# 3007 dims at N=1000. --chains 8 vectorised over the single GPU.
for N in 100 1000; do
    echo; echo "=== N=$N, GPU ==="
    $ENV/bin/python poc/rung3_hierarchical.py --nlens $N --chains 8 --device gpu
done

# is the GPU actually helping? we run in float64 and the A40 does FP64 at 1/32 rate,
# on arrays of ~50k elements. Compare against the 8 CPU cores of the same node.
echo; echo "=== N=1000, CPU on the same node, for comparison ==="
$ENV/bin/python poc/rung3_hierarchical.py --nlens 1000 --chains 8 --device cpu

echo; echo "=== parameterisation comparison (N=1000, GPU) ==="
for P in mixed noncentred centred; do
    $ENV/bin/python poc/rung3_hierarchical.py --nlens 1000 --chains 4 --param $P --device gpu --short
done
