#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LeWM (Web) — Login Node Setup for IITM Para / Rudra HPCE
# Run this ONCE on the login node (has internet access).
# Do NOT run inside a SLURM job.
#
# Usage:
#   bash hpce/setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

export SCRATCH="/scratch/$USER"
REPO_DIR="$SCRATCH/le-wm"
SPT_DIR="$SCRATCH/stable-pretraining"
CONDA_ENV="$SCRATCH/lewm-conda"
DATASET_DIR="$SCRATCH/datasets"
CONDA=/home/apps/miniconda3/bin/conda

echo "============================================================"
echo "Setting up LeWM (Web) on IITM HPCE"
echo "User    : $USER"
echo "Scratch : $SCRATCH"
echo "Repo    : $REPO_DIR"
echo "Env     : $CONDA_ENV"
echo "Dataset : $DATASET_DIR"
echo "============================================================"

mkdir -p "$DATASET_DIR" "$SCRATCH/logs"

module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true

# ── Conda environment (Python 3.11 with all C extensions) ────────────────────
if [ ! -d "$CONDA_ENV" ]; then
    echo "Creating conda env with Python 3.11..."
    $CONDA create -y -p "$CONDA_ENV" python=3.11
fi

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "Python  : $(python --version)"

pip install --upgrade pip -q

# Core training deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install lightning transformers -q
pip install hydra-core omegaconf -q
pip install lancedb pylance pyarrow==20.0.0 -q
pip install wandb loguru prettytable richuru tabulate -q
pip install opencv-python-headless imageio pandas zstandard requests-cache -q
pip install submitit -q

# stable-worldmodel from PyPI
pip install stable-worldmodel==0.0.6 -q

# stable-pretraining from GitHub (our fork with fixes)
if [ ! -d "$SPT_DIR" ]; then
    git clone https://github.com/Anurag9Dhiman/stable-pretraining.git "$SPT_DIR"
    echo "Cloned stable-pretraining"
else
    echo "Pulling stable-pretraining..."
    git -C "$SPT_DIR" pull
fi
pip install -e "$SPT_DIR" -q

# ── Clone / update le-wm repo ─────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Anurag9Dhiman/le-wm.git "$REPO_DIR"
    echo "Cloned le-wm"
else
    echo "Pulling le-wm..."
    git -C "$REPO_DIR" pull
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Next: upload dataset from your Mac:"
echo "  rsync -avz --progress /Users/\$USER/Documents/LeWM/datasets/openapps_all.lance \\"
echo "      \$USER@hpce.iitm.ac.in:$DATASET_DIR/"
echo ""
echo "Then submit the job:"
echo "  sbatch $REPO_DIR/hpce/train.slurm"
echo "============================================================"
