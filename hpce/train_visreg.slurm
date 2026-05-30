#!/bin/bash
#SBATCH --job-name=lewm-visreg
#SBATCH --output=/scratch/%u/logs/lewm_visreg_%j.log
#SBATCH --error=/scratch/%u/logs/lewm_visreg_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ─────────────────────────────────────────────────────────────────────────────
# LeWM (Web) — VISReg Training Job
# VISReg uses 2x higher LR and weight=4.5 vs SIGReg weight=0.05
# Submit: cd /scratch/$USER/le-wm && sbatch hpce/train_visreg.slurm
# Logs  : /scratch/$USER/logs/lewm_visreg_<jobid>.log
# ─────────────────────────────────────────────────────────────────────────────

export SCRATCH="/scratch/$USER"
REPO_DIR="$SCRATCH/le-wm"
CONDA_ENV="$SCRATCH/lewm-conda"

mkdir -p "$SCRATCH/logs" "$SCRATCH/checkpoints/web_lewm_visreg"

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"

module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export LOCAL_DATASET_DIR="$SCRATCH/datasets"
export SCRATCH="$SCRATCH"
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true

cd "$REPO_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Starting training (VISReg)..."
python train.py \
    --config-name web_lewm \
    output_model_name=web_lewm_visreg \
    optimizer.lr=6e-5 \
    loss.sigreg.weight=0.0 \
    loss.visreg.weight=4.5 \
    loader.num_workers=$SLURM_CPUS_PER_TASK \
    "hydra.run.dir=$SCRATCH/hydra_outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"

echo "Finished: $(date)"
