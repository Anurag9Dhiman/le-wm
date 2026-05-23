#!/bin/bash
#SBATCH --job-name=lewm-viz
#SBATCH --output=/scratch/%u/logs/lewm_viz_%j.log
#SBATCH --error=/scratch/%u/logs/lewm_viz_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ─────────────────────────────────────────────────────────────────────────────
# LeWM Visualization Job — saves GIFs to /scratch/$USER/stablewm/videos/
# Transfer to local: scp username@para.hpce.iitm.ac.in:/scratch/username/stablewm/videos/*.gif .
# Submit: sbatch hpce/visualize.slurm
# ─────────────────────────────────────────────────────────────────────────────

TASKS="tworoom pusht"
N_STEPS=80

SCRATCH="/scratch/$USER"
REPO_DIR="$SCRATCH/le-wm"
STABLEWM_HOME="$SCRATCH/stablewm"
VENV="$SCRATCH/lewm-env"
VIDEO_DIR="$STABLEWM_HOME/videos"

mkdir -p "$VIDEO_DIR" "$SCRATCH/logs"

module load python/3.10   2>/dev/null || module load python3/3.10 2>/dev/null || true
module load cuda/12.1     2>/dev/null || module load cuda/11.8    2>/dev/null || true

source "$VENV/bin/activate"

export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

cd "$REPO_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 - <<PYEOF
import os, sys, json, torch, imageio
from pathlib import Path
from torchvision.transforms import v2 as T
import stable_pretraining as spt
import stable_worldmodel as swm
from jepa import JEPA
from module import ARPredictor, Embedder, MLP

STABLEWM_HOME = Path(os.environ["STABLEWM_HOME"])
VIDEO_DIR     = STABLEWM_HOME / "videos"
N_STEPS       = $N_STEPS
device        = "cuda" if torch.cuda.is_available() else "cpu"

ENV_NAMES = {
    "tworoom": "swm/TwoRoom-v1",
    "pusht":   "swm/PushT-v1",
}

normalize = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Resize(224),
])

def hydra_kwargs(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}

def make_mlp(cfg, k):
    return MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
               hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)

def load_model(task):
    ckpt = STABLEWM_HOME / task / "lewm_object.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Run train.slurm first.")
    model = torch.load(ckpt, map_location=device, weights_only=False)
    return model.to(device).eval()

tasks = "$TASKS".split()

for task in tasks:
    print(f"\n{'='*50}")
    print(f"Task: {task.upper()}")

    try:
        model = load_model(task)
        print(f"  Loaded: {sum(p.numel() for p in model.parameters()):,} params")

        solver = swm.solver.CEMSolver(
            model=model, batch_size=1, num_samples=50,
            var_scale=1.0, n_steps=5, topk=5, device=device, seed=42,
        )
        plan_cfg = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
        policy   = swm.policy.WorldModelPolicy(
            solver=solver, config=plan_cfg, process={},
            transform={"pixels": normalize, "goal": normalize},
        )

        world = swm.World(env_name=ENV_NAMES[task], num_envs=1,
                          max_episode_steps=N_STEPS + 10, image_shape=(224, 224))
        world.set_policy(policy)

        frames = []
        obs, _ = world.reset()
        for step in range(N_STEPS):
            frame = world.render()
            if frame is not None:
                frames.append(frame)
            action = world.get_action(obs)
            obs, _, done, _, _ = world.step(action)
            if done.all():
                break

        gif_path = VIDEO_DIR / f"{task}_lewm.gif"
        imageio.mimsave(str(gif_path), frames, fps=10, loop=0)
        print(f"  Saved: {gif_path}  ({len(frames)} frames)")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

        # fallback: random policy
        try:
            world = swm.World(env_name=ENV_NAMES[task], num_envs=1,
                              max_episode_steps=N_STEPS + 10, image_shape=(224, 224))
            world.set_policy(swm.policy.RandomPolicy())
            frames, obs = [], world.reset()[0]
            for _ in range(N_STEPS):
                frame = world.render()
                if frame is not None: frames.append(frame)
                obs, _, done, _, _ = world.step(world.get_action(obs))
                if done.all(): break
            gif_path = VIDEO_DIR / f"{task}_random.gif"
            imageio.mimsave(str(gif_path), frames, fps=10, loop=0)
            print(f"  Fallback saved: {gif_path}")
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")

print(f"\nAll GIFs saved to: {VIDEO_DIR}")
print("Transfer to local machine:")
print(f"  scp $USER@para.hpce.iitm.ac.in:{VIDEO_DIR}/*.gif ~/Desktop/")
PYEOF

echo "Visualization done: $(date)"
