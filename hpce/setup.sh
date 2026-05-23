#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LeWM — Login Node Setup for IITM Para / Rudra HPCE
# Run this ONCE on the login node (has internet access).
# Do NOT run inside a SLURM job.
#
# Usage:
#   bash hpce/setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRATCH="/scratch/$USER"
REPO_DIR="$SCRATCH/le-wm"
STABLEWM_HOME="$SCRATCH/stablewm"
VENV="$SCRATCH/lewm-env"

mkdir -p "$STABLEWM_HOME"

echo "============================================================"
echo "Setting up LeWM on IITM HPCE"
echo "User       : $USER"
echo "Scratch    : $SCRATCH"
echo "Repo       : $REPO_DIR"
echo "Data/ckpts : $STABLEWM_HOME"
echo "Venv       : $VENV"
echo "============================================================"

# ── Load modules (adjust names with: module avail python, module avail cuda) ──
module load python/3.10   2>/dev/null || module load python3/3.10 2>/dev/null || true
module load cuda/12.1     2>/dev/null || module load cuda/11.8    2>/dev/null || true
echo "Python: $(python3 --version)"

# ── Python virtual environment ────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "Created venv at $VENV"
fi
source "$VENV/bin/activate"

pip install --upgrade pip -q
pip install stable-worldmodel[train,env] -q
pip install zstandard imageio[ffmpeg] Pillow -q

echo "Dependencies installed."

# ── Clone repo ────────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/lucas-maes/le-wm.git "$REPO_DIR"
    echo "Repo cloned to $REPO_DIR"
else
    echo "Repo already exists. Pulling latest..."
    git -C "$REPO_DIR" pull
fi

# ── Download datasets ─────────────────────────────────────────────────────────
echo ""
echo "Downloading datasets from HuggingFace..."

pip install huggingface_hub -q

python3 - <<'PY'
import os, subprocess, shutil, tarfile, zstandard as zstd, io
from pathlib import Path
from huggingface_hub import snapshot_download

STABLEWM_HOME = Path(os.environ["SCRATCH"]) / "stablewm"

datasets = {
    "tworoom": ("quentinll/lewm-tworooms", "tworoom.h5"),
    "pusht":   ("quentinll/lewm-pusht",    "pusht_expert_train.lance"),
}

for name, (repo, filename) in datasets.items():
    dst = STABLEWM_HOME / filename
    if dst.exists():
        print(f"  {filename} already present.")
        continue
    tmp = STABLEWM_HOME / f"hf_{name}"
    print(f"  Downloading {repo} ...")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(tmp))

    zsts = list(tmp.rglob("*.tar.zst")) + list(tmp.rglob("*.zst"))
    if zsts:
        zst = zsts[0]
        print(f"  Decompressing {zst.name} ...")
        ret = subprocess.run(f"zstd -d -c '{zst}' | tar -x -C '{STABLEWM_HOME}'", shell=True)
        if ret.returncode != 0:
            print(f"  Warning: pipe failed, trying Python fallback...")
            with open(zst, "rb") as f:
                raw = zstd.ZstdDecompressor().decompress(f.read(), max_output_size=50*1024**3)
            with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                tar.extractall(STABLEWM_HOME, filter="data")
        zst.unlink()
        shutil.rmtree(tmp, ignore_errors=True)

    found = list(STABLEWM_HOME.rglob(f"*{Path(filename).suffix}"))
    if found:
        if str(found[0]) != str(dst):
            shutil.move(str(found[0]), str(dst))
        print(f"  Ready: {dst}")
    else:
        print(f"  WARNING: could not find {filename} after extraction.")

print("Dataset setup complete.")
PY

# ── Download pretrained checkpoints ──────────────────────────────────────────
echo ""
echo "Downloading pretrained checkpoints..."

python3 - <<'PY'
import os, json, torch
from pathlib import Path
from huggingface_hub import snapshot_download

STABLEWM_HOME = Path(os.environ["SCRATCH"]) / "stablewm"
REPO_DIR      = Path(os.environ["SCRATCH"]) / "le-wm"

import sys
sys.path.insert(0, str(REPO_DIR))

import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor, Embedder, MLP

def hydra_kwargs(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}

def make_mlp(cfg, k):
    return MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
               hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)

tasks = {
    "tworoom": "quentinll/lewm-tworooms",
    "pusht":   "quentinll/lewm-pusht",
}

for task, repo in tasks.items():
    ckpt = STABLEWM_HOME / task / "lewm_object.ckpt"
    if ckpt.exists():
        print(f"  {task}: checkpoint already present.")
        continue
    hf_local = STABLEWM_HOME / f"hf_{task}"
    print(f"  Downloading {repo} ...")
    snapshot_download(repo, local_dir=str(hf_local))

    cfg = json.loads((hf_local / "config.json").read_text())
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"], patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"], pretrained=False, use_mask_token=False,
    )
    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**hydra_kwargs(cfg["predictor"])),
        action_encoder=Embedder(**hydra_kwargs(cfg["action_encoder"])),
        projector=make_mlp(cfg, "projector"),
        pred_proj=make_mlp(cfg, "pred_proj"),
    )
    sd = torch.load(hf_local / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, ckpt)
    print(f"  Saved: {ckpt}")

print("Checkpoints ready.")
PY

echo ""
echo "============================================================"
echo "Setup complete. Submit jobs with:"
echo "  sbatch hpce/train.slurm"
echo "  sbatch hpce/visualize.slurm"
echo "============================================================"
