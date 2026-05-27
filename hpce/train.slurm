#!/bin/bash
#SBATCH --job-name=lewm-web
#SBATCH --output=/scratch/%u/logs/lewm_web_%j.log
#SBATCH --error=/scratch/%u/logs/lewm_web_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# Adjust partition to your cluster:
#   Para  → --partition=gpu  or  --partition=standard
#   Rudra → --partition=gpu  or  --partition=a100
#SBATCH --partition=gpu

# ─────────────────────────────────────────────────────────────────────────────
# LeWM Web Training Job — IITM Para / Rudra HPCE
# Submit: sbatch hpce/train.slurm
# Logs  : /scratch/$USER/logs/lewm_web_<jobid>.log
# ─────────────────────────────────────────────────────────────────────────────

MAX_EPOCHS=100
BATCH_SIZE=128        # A100 can handle 128; drop to 64 for V100

export SCRATCH="/scratch/$USER"
REPO_DIR="$SCRATCH/le-wm"
CONDA_ENV="$SCRATCH/lewm-conda"
DATASET_DIR="$SCRATCH/datasets"

mkdir -p "$SCRATCH/logs"

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Epochs   : $MAX_EPOCHS"
echo "Batch    : $BATCH_SIZE"
echo "Started  : $(date)"

# ── Load modules ──────────────────────────────────────────────────────────────
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export LOCAL_DATASET_DIR="$DATASET_DIR"
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true    # set to false and add WANDB_API_KEY to use W&B

cd "$REPO_DIR"

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Patch config: epochs, batch size, workers ─────────────────────────────────
python - <<PYEOF
import yaml
from pathlib import Path

cfg_path = Path("config/train/web_lewm.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["trainer"]["max_epochs"]        = $MAX_EPOCHS
cfg["trainer"]["precision"]         = "bf16-mixed"
cfg["loader"]["batch_size"]         = $BATCH_SIZE
cfg["loader"]["num_workers"]        = $SLURM_CPUS_PER_TASK
cfg["loader"]["prefetch_factor"]    = 4
cfg["loader"]["persistent_workers"] = True
cfg["loader"]["pin_memory"]         = True

cfg_path.write_text(yaml.dump(cfg, default_flow_style=False))
print(f"Config patched: epochs={cfg['trainer']['max_epochs']}, "
      f"batch={cfg['loader']['batch_size']}, "
      f"workers={cfg['loader']['num_workers']}, "
      f"precision={cfg['trainer']['precision']}")
PYEOF

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Starting training..."
python train.py \
    --config-name web_lewm \
    "hydra.run.dir=$SCRATCH/hydra_outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"

# ── Copy checkpoint to named location ────────────────────────────────────────
python - <<PYEOF
import shutil
from pathlib import Path
import stable_worldmodel as swm

ckpt_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
matches = sorted(ckpt_dir.glob("**/web_lewm_object.ckpt"), key=lambda p: p.stat().st_mtime)
if matches:
    src = matches[-1]
    dst = Path("$SCRATCH") / "web_lewm_object.ckpt"
    shutil.copy(src, dst)
    print(f"Checkpoint saved: {dst}")
else:
    print("WARNING: no checkpoint found in", ckpt_dir)
PYEOF

echo "Finished: $(date)"
