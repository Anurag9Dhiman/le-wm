"""Trajectory recorder for the Web-World-Models AI Alchemy app.

Collects (screenshot, action) pairs using Playwright and saves as a LanceDB
table compatible with stable_worldmodel's LanceDataset loader.

Usage:
    # 1. Start the AI Alchemy dev server in another terminal:
    #    cd /path/to/Web-World-Models/src/AI_ALCHEMY && npm install && npm run dev

    # 2. Run the collector:
    python data/collect_ai_alchemy.py --output /path/to/ai_alchemy_train.lance

    # 3. Point stable_worldmodel at the output directory:
    #    export LOCAL_DATASET_DIR=/path/to/output_dir
    #    python train.py data=web

Dependencies:
    pip install playwright lancedb pyarrow pillow
    playwright install chromium
"""

import argparse
import asyncio
import io
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

try:
    import lancedb
except ImportError:
    print("lancedb not found. Install with: pip install lancedb")
    sys.exit(1)

try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    print("playwright not found. Install with: pip install playwright && playwright install chromium")
    sys.exit(1)

# Add repo root to path so we can import web_action
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.web_action import encode_action, ACTION_DIM, ACTION_TYPES

IMG_SIZE = 224  # must match config/train/web_lewm.yaml img_size


# ──────────────────────────────────────────────────────────────────────────────
# LanceDB schema
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA = pa.schema([
    pa.field("pixels", pa.list_(pa.uint8(), IMG_SIZE * IMG_SIZE * 3)),
    pa.field("action", pa.list_(pa.float32(), ACTION_DIM)),
    pa.field("episode_idx", pa.int32()),
    pa.field("step_idx", pa.int32()),
])


# ──────────────────────────────────────────────────────────────────────────────
# Screenshot helper
# ──────────────────────────────────────────────────────────────────────────────

async def capture_screenshot(page: Page) -> np.ndarray:
    """Return (H, W, 3) uint8 array at IMG_SIZE resolution."""
    png = await page.screenshot(type="png")
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# AI Alchemy app interaction
# ──────────────────────────────────────────────────────────────────────────────

ELEMENTS = [
    "sand", "water", "fire", "stone", "wood", "oil", "lava", "ice",
    "steam", "salt", "plant", "acid", "metal", "glass", "void", "clone", "air",
]

# Selectors to try for the simulation canvas (order = priority)
CANVAS_SELECTORS = [
    "canvas",
    "[class*='canvas']",
    "[class*='Canvas']",
    "[class*='simulation']",
    "[class*='grid']",
]

# Selectors to try for element picker buttons
ELEMENT_BTN_SELECTORS = [
    "[class*='element']",
    "[class*='Element']",
    "[class*='tool']",
    "[class*='material']",
    "button",
]


async def find_canvas(page: Page) -> dict | None:
    """Return bounding box dict or None if no canvas found."""
    for sel in CANVAS_SELECTORS:
        el = await page.query_selector(sel)
        if el:
            bb = await el.bounding_box()
            if bb and bb["width"] > 50 and bb["height"] > 50:
                return bb
    return None


async def select_element(page: Page, element_name: str):
    """Try to select an element by clicking its button."""
    for sel in ELEMENT_BTN_SELECTORS:
        els = await page.query_selector_all(sel)
        for el in els:
            text = (await el.inner_text()).strip().lower()
            if element_name in text:
                await el.click()
                return
    # fallback: press keyboard shortcut 1-9
    idx = ELEMENTS.index(element_name) if element_name in ELEMENTS else 0
    await page.keyboard.press(str((idx % 9) + 1))


async def paint_stroke(page: Page, bb: dict) -> tuple[float, float]:
    """Mouse-down drag across the canvas. Returns (x_norm, y_norm) of start."""
    margin = max(5.0, bb["width"] * 0.05)
    x1 = random.uniform(bb["x"] + margin, bb["x"] + bb["width"] - margin)
    y1 = random.uniform(bb["y"] + margin, bb["y"] + bb["height"] - margin)
    dx = random.uniform(-bb["width"] * 0.25, bb["width"] * 0.25)
    dy = random.uniform(-bb["height"] * 0.25, bb["height"] * 0.25)
    x2 = float(np.clip(x1 + dx, bb["x"] + margin, bb["x"] + bb["width"] - margin))
    y2 = float(np.clip(y1 + dy, bb["y"] + margin, bb["y"] + bb["height"] - margin))

    await page.mouse.move(x1, y1)
    await page.mouse.down()
    steps = random.randint(4, 12)
    for i in range(1, steps + 1):
        xi = x1 + (x2 - x1) * i / steps
        yi = y1 + (y2 - y1) * i / steps
        await page.mouse.move(xi, yi)
    await page.mouse.up()

    x_norm = (x1 - bb["x"]) / bb["width"]
    y_norm = (y1 - bb["y"]) / bb["height"]
    return x_norm, y_norm


async def single_click(page: Page, bb: dict) -> tuple[float, float]:
    """Single click on a random canvas position."""
    margin = max(5.0, bb["width"] * 0.05)
    px = random.uniform(bb["x"] + margin, bb["x"] + bb["width"] - margin)
    py = random.uniform(bb["y"] + margin, bb["y"] + bb["height"] - margin)
    await page.mouse.click(px, py)
    return (px - bb["x"]) / bb["width"], (py - bb["y"]) / bb["height"]


# ──────────────────────────────────────────────────────────────────────────────
# Episode collection
# ──────────────────────────────────────────────────────────────────────────────

async def collect_episode(page: Page, episode_len: int = 64) -> list[dict]:
    """Collect one episode; returns list of {pixels, action} dicts."""
    bb = await find_canvas(page)
    if bb is None:
        # No canvas found — use full viewport as fallback
        vp = page.viewport_size
        bb = {"x": 0, "y": 0, "width": vp["width"], "height": vp["height"]}

    steps = []
    current_element = random.choice(ELEMENTS)
    await select_element(page, current_element)

    for _ in range(episode_len):
        pixels = await capture_screenshot(page)

        # Sample action type with realistic distribution
        choice = random.choices(
            ["paint", "click", "change_element", "noop"],
            weights=[0.50, 0.25, 0.20, 0.05],
        )[0]

        if choice == "paint":
            x_n, y_n = await paint_stroke(page, bb)
            action_vec = encode_action("drag", x_n, y_n, text=current_element)

        elif choice == "click":
            x_n, y_n = await single_click(page, bb)
            action_vec = encode_action("click", x_n, y_n, text=current_element)

        elif choice == "change_element":
            current_element = random.choice(ELEMENTS)
            await select_element(page, current_element)
            action_vec = encode_action("click", 0.0, 0.0, text=current_element)

        else:
            action_vec = encode_action("noop")

        steps.append({"pixels": pixels, "action": action_vec})
        await asyncio.sleep(0.05)

    return steps


# ──────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ──────────────────────────────────────────────────────────────────────────────

async def collect(
    app_url: str,
    output_dir: str,
    table_name: str,
    num_episodes: int,
    episode_len: int,
    headless: bool,
    write_interval: int = 10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(output_dir))
    table = None
    pending: list[dict] = []
    total_steps = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await ctx.new_page()

        print(f"→ Navigating to {app_url}")
        await page.goto(app_url, wait_until="networkidle", timeout=30_000)
        await asyncio.sleep(2.0)

        for ep_idx in range(num_episodes):
            print(f"  Episode {ep_idx + 1:4d}/{num_episodes} ...", end="", flush=True)

            await page.reload(wait_until="networkidle")
            await asyncio.sleep(1.0)

            steps = await collect_episode(page, episode_len=episode_len)

            for s_idx, step in enumerate(steps):
                pending.append({
                    "pixels": step["pixels"].flatten().tolist(),
                    "action": step["action"].tolist(),
                    "episode_idx": ep_idx,
                    "step_idx": s_idx,
                })

            total_steps += len(steps)
            print(f" {len(steps)} steps (total {total_steps})")

            if (ep_idx + 1) % write_interval == 0 or (ep_idx + 1) == num_episodes:
                batch = pa.Table.from_pylist(pending, schema=SCHEMA)
                if table is None:
                    table = db.create_table(table_name, batch, mode="overwrite")
                else:
                    table.add(batch)
                pending = []
                print(f"  ✓ Flushed to {output_dir / (table_name + '.lance')}")

        await browser.close()

    print(f"\nDone. {num_episodes} episodes × {episode_len} steps = {total_steps} rows")
    print(f"Output: {output_dir / (table_name + '.lance')}")
    print(f"\nTo train: export LOCAL_DATASET_DIR={output_dir}")
    print(f"          python train.py --config-name web_lewm data.dataset.name={table_name}.lance")


# ──────────────────────────────────────────────────────────────────────────────
# Optional: start the dev server automatically
# ──────────────────────────────────────────────────────────────────────────────

def start_dev_server(app_dir: str, port: int = 3000) -> subprocess.Popen:
    """Start 'npm run dev' in app_dir as a background process."""
    proc = subprocess.Popen(
        ["npm", "run", "dev", "--", f"--port={port}"],
        cwd=app_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Started dev server (pid {proc.pid}) — waiting for it to come up...")
    time.sleep(6)  # give Vite/webpack time to compile
    return proc


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Collect AI Alchemy trajectories")
    p.add_argument("--url", default="http://localhost:3000",
                   help="URL of the running AI Alchemy app")
    p.add_argument("--app-dir", default=None,
                   help="If given, starts 'npm run dev' in this directory before recording")
    p.add_argument("--output-dir", default="datasets",
                   help="Directory to write the LanceDB table into")
    p.add_argument("--table-name", default="ai_alchemy_train",
                   help="LanceDB table name (without .lance extension)")
    p.add_argument("--num-episodes", type=int, default=200,
                   help="Number of episodes to collect")
    p.add_argument("--episode-len", type=int, default=64,
                   help="Steps per episode")
    p.add_argument("--no-headless", action="store_true",
                   help="Show browser window during collection")
    p.add_argument("--write-interval", type=int, default=10,
                   help="Flush to disk every N episodes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dev_server = None
    if args.app_dir:
        dev_server = start_dev_server(args.app_dir)

    try:
        asyncio.run(collect(
            app_url=args.url,
            output_dir=args.output_dir,
            table_name=args.table_name,
            num_episodes=args.num_episodes,
            episode_len=args.episode_len,
            headless=not args.no_headless,
            write_interval=args.write_interval,
        ))
    finally:
        if dev_server is not None:
            dev_server.terminate()
