#!/bin/bash
#SBATCH --job-name=lewm-train
#SBATCH --output=/scratch/%u/logs/lewm_train_%j.log
#SBATCH --error=/scratch/%u/logs/lewm_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# Adjust partition to your cluster:
#   Para  → --partition=gpu  or  --partition=standard
#   Rudra → --partition=gpu  or  --partition=a100
#SBATCH --partition=gpu

# ─────────────────────────────────────────────────────────────────────────────
# LeWM Training Job — IITM Para / Rudra HPCE
# Submit: sbatch hpce/train.slurm
# Logs  : /scratch/$USER/logs/lewm_train_<jobid>.log
# ─────────────────────────────────────────────────────────────────────────────

TASK="tworoom"        # "tworoom" | "pusht"
MAX_EPOCHS=100
BATCH_SIZE=128        # A100 can handle 128; drop to 64 for V100

SCRATCH="/scratch/$USER"
REPO_DIR="$SCRATCH/le-wm"
STABLEWM_HOME="$SCRATCH/stablewm"
VENV="$SCRATCH/lewm-env"

mkdir -p "$SCRATCH/logs"

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Task     : $TASK"
echo "Epochs   : $MAX_EPOCHS"
echo "Started  : $(date)"

# ── Load modules ──────────────────────────────────────────────────────────────
module load python/3.10   2>/dev/null || module load python3/3.10 2>/dev/null || true
module load cuda/12.1     2>/dev/null || module load cuda/11.8    2>/dev/null || true

source "$VENV/bin/activate"

export STABLEWM_HOME="$STABLEWM_HOME"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export MUJOCO_GL="egl"
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true    # set to false and add WANDB_API_KEY to use W&B

cd "$REPO_DIR"

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Patch train.py if needed (load_dataset → HDF5Dataset) ────────────────────
python3 - <<'PY'
from pathlib import Path
train_py = Path("train.py")
src = train_py.read_text()
old = (
    "    dataset = swm.data.load_dataset(\n"
    "        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg\n"
    "    )"
)
new = (
    "    dataset_name_bare = dataset_name.rsplit('.', 1)[0]\n"
    "    dataset = swm.data.HDF5Dataset(\n"
    "        name=dataset_name_bare,\n"
    "        frameskip=dataset_cfg.pop('frameskip', 5),\n"
    "        num_steps=dataset_cfg.pop('num_steps', 4),\n"
    "        transform=None,\n"
    "        keys_to_load=dataset_cfg.pop('keys_to_load', None),\n"
    "        keys_to_cache=dataset_cfg.pop('keys_to_cache', None),\n"
    "        keys_to_merge=None,\n"
    "        cache_dir=cache_dir,\n"
    "    )"
)
if old in src:
    train_py.write_text(src.replace(old, new))
    print("train.py patched.")
elif "HDF5Dataset" in src:
    print("train.py already patched.")
PY

# ── Patch config: precision + workers ────────────────────────────────────────
python3 - <<PYEOF
import yaml, torch
from pathlib import Path

cfg_path = Path("config/train/lewm.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
cfg["trainer"]["precision"]         = "bf16-mixed" if supports_bf16 else "16-mixed"
cfg["trainer"]["max_epochs"]        = $MAX_EPOCHS
cfg["loader"]["batch_size"]         = $BATCH_SIZE
cfg["loader"]["num_workers"]        = $SLURM_CPUS_PER_TASK
cfg["loader"]["prefetch_factor"]    = 4
cfg["loader"]["persistent_workers"] = True
cfg["wandb"]                        = {"enabled": False, "config": {}}

cfg_path.write_text(yaml.dump(cfg, default_flow_style=False))
print(f"Config patched: precision={cfg['trainer']['precision']}, workers={cfg['loader']['num_workers']}")
PYEOF

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Starting training..."
python3 train.py \
    data=$TASK \
    "hydra.run.dir=$STABLEWM_HOME/hydra_outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"

# ── Copy checkpoint to expected location ──────────────────────────────────────
python3 - <<PYEOF
import shutil
from pathlib import Path

ckpt_dir = Path("$STABLEWM_HOME") / "checkpoints"
matches = sorted(ckpt_dir.glob("*/lewm_object.ckpt"), key=lambda p: p.stat().st_mtime)
if matches:
    src = matches[-1]
    dst = Path("$STABLEWM_HOME") / "$TASK" / "lewm_object.ckpt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"Checkpoint saved: {dst}")
else:
    print("WARNING: no checkpoint found.")
PYEOF

echo "Finished: $(date)"
