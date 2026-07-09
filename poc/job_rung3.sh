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
export MPLCONFIGDIR=$TMPDIR/mpl

echo "=== node $(hostname)  job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

$ENV/bin/python - <<'PY'
import jax
print("jax", jax.__version__, "| devices:", jax.devices())
PY

# 3007 dims at N=1000. --chains 8 vectorised over the single GPU.
for N in 100 1000; do
    echo; echo "=== N=$N ==="
    $ENV/bin/python poc/rung3_hierarchical.py --nlens $N --chains 8 --device gpu
done

echo; echo "=== parameterisation comparison (N=1000) ==="
for P in mixed noncentred centred; do
    $ENV/bin/python poc/rung3_hierarchical.py --nlens 1000 --chains 4 --param $P --device gpu --short
done
