"""Trajectory recorder for OpenApps.

Launches the OpenApps server (or connects to a running one), then uses
Playwright to perform random interactions across the sub-apps, saving
(screenshot, action) pairs as a LanceDB table.

Usage:
    # Option A – auto-launch (requires OpenApps repo):
    python data/collect_open_apps.py \\
        --open-apps-dir /path/to/OpenApps \\
        --apps todo calendar messenger \\
        --num-episodes 200 --episode-len 48

    # Option B – connect to already-running server:
    python data/collect_open_apps.py \\
        --base-url http://localhost:5001 \\
        --apps todo calendar \\
        --num-episodes 200 --episode-len 48

Output:
    datasets/openapps_<app>.lance  per app, or one merged table.
    Set LOCAL_DATASET_DIR=datasets and train with:
        python train.py --config-name web_lewm data.dataset.name=openapps_all.lance

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
    print("lancedb not found: pip install lancedb"); sys.exit(1)
try:
    from playwright.async_api import async_playwright, Page, Browser
except ImportError:
    print("playwright not found: pip install playwright && playwright install chromium"); sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.web_action import encode_action, ACTION_DIM

IMG_SIZE = 224  # must match config/train/web_lewm.yaml img_size

SCHEMA = pa.schema([
    pa.field("pixels", pa.list_(pa.uint8(), IMG_SIZE * IMG_SIZE * 3)),
    pa.field("action",  pa.list_(pa.float32(), ACTION_DIM)),
    pa.field("episode_idx", pa.int32()),
    pa.field("step_idx",    pa.int32()),
])

# ──────────────────────────────────────────────────────────────────────────────
# App route map  (relative to base_url)
# ──────────────────────────────────────────────────────────────────────────────

APP_ROUTES: dict[str, str] = {
    "todo":       "/",
    "calendar":   "/calendar",
    "messenger":  "/messages",
    "codeeditor": "/codeeditor",
    "onlineshop": "/onlineshop",
    "map":        "/maps",
}

# Sample text strings that make sense across all apps
TEXT_POOL = [
    "meeting with team", "buy groceries", "review PR", "call client",
    "lunch at noon",  "dentist appointment", "hello", "test",
    "10:30 AM", "2024-06-15", "New York", "confirm", "urgent",
    "fix bug", "deploy to prod", "design review",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def screenshot_arr(page: Page) -> np.ndarray:
    """Capture page screenshot → (H, W, 3) uint8 at IMG_SIZE."""
    raw = await page.screenshot(type="png")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


async def interactive_elements(page: Page) -> list[dict]:
    """Return visible interactive elements with bounding boxes."""
    selectors = [
        "button:visible",
        "input:visible",
        "textarea:visible",
        "select:visible",
        "a[href]:visible",
        "[role='button']:visible",
        "[contenteditable='true']:visible",
    ]
    found = []
    for sel in selectors:
        for el in await page.query_selector_all(sel):
            try:
                bb = await el.bounding_box()
                if not bb or bb["width"] < 5 or bb["height"] < 5:
                    continue
                tag  = await el.evaluate("e => e.tagName.toLowerCase()")
                kind = (await el.get_attribute("type") or "").lower()
                found.append({"el": el, "bb": bb, "tag": tag, "kind": kind})
            except Exception:
                pass
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Episode collection
# ──────────────────────────────────────────────────────────────────────────────

async def collect_episode(
    page: Page,
    episode_len: int,
    viewport_w: int,
    viewport_h: int,
) -> list[dict]:
    """Run one episode of random interactions; return list of {pixels, action}."""
    steps = []

    for _ in range(episode_len):
        pixels = await screenshot_arr(page)

        els = await interactive_elements(page)

        if not els or random.random() < 0.04:
            steps.append({"pixels": pixels, "action": encode_action("noop")})
            await asyncio.sleep(0.08)
            continue

        item = random.choice(els)
        bb   = item["bb"]
        cx   = bb["x"] + bb["width"]  / 2
        cy   = bb["y"] + bb["height"] / 2
        x_n  = cx / viewport_w
        y_n  = cy / viewport_h

        if item["tag"] in ("input", "textarea") and item["kind"] not in ("checkbox", "radio", "submit", "button", "file", "date", "time"):
            # text entry
            text = random.choice(TEXT_POOL)
            try:
                await item["el"].triple_click()
                await item["el"].type(text, delay=25)
            except Exception:
                await page.mouse.click(cx, cy)
            action_vec = encode_action("type", x_n, y_n, text=text[:15])

        elif item["tag"] == "select":
            options = await item["el"].evaluate(
                "el => Array.from(el.options, o => o.value).filter(v => v)"
            )
            if options:
                await item["el"].select_option(random.choice(options))
            action_vec = encode_action("click", x_n, y_n)

        elif item["kind"] in ("date", "time"):
            # fill date/time inputs
            val = "2024-06-15" if item["kind"] == "date" else "10:30"
            try:
                await item["el"].fill(val)
            except Exception:
                pass
            action_vec = encode_action("type", x_n, y_n, text=val)

        else:
            # plain click (button, link, checkbox, radio, etc.)
            try:
                await page.mouse.click(cx, cy)
            except Exception:
                pass
            action_vec = encode_action("click", x_n, y_n)

        steps.append({"pixels": pixels, "action": action_vec})
        await asyncio.sleep(0.12)

    return steps


# ──────────────────────────────────────────────────────────────────────────────
# Per-app collection loop
# ──────────────────────────────────────────────────────────────────────────────

async def collect_app(
    browser: Browser,
    base_url: str,
    app_name: str,
    route: str,
    num_episodes: int,
    episode_len: int,
    db: "lancedb.DBConnection",
    table_name: str,
    viewport: tuple[int, int],
    write_interval: int,
) -> int:
    """Collect trajectories for one app; returns total steps written."""
    vw, vh = viewport
    ctx  = await browser.new_context(viewport={"width": vw, "height": vh})
    page = await ctx.new_page()
    url  = f"{base_url.rstrip('/')}{route}"

    print(f"\n  [{app_name}] navigating to {url}")
    await page.goto(url, wait_until="networkidle", timeout=30_000)
    await asyncio.sleep(1.5)

    pending: list[dict] = []
    table = None
    total = 0

    for ep_idx in range(num_episodes):
        print(f"    ep {ep_idx + 1:3d}/{num_episodes}", end="", flush=True)

        # reload to reset app state between episodes
        await page.reload(wait_until="networkidle")
        await asyncio.sleep(0.8)

        steps = await collect_episode(page, episode_len, vw, vh)

        for s_idx, step in enumerate(steps):
            pending.append({
                "pixels":      step["pixels"].flatten().tolist(),
                "action":      step["action"].tolist(),
                "episode_idx": ep_idx,
                "step_idx":    s_idx,
            })

        total += len(steps)
        print(f"  {len(steps)} steps")

        if (ep_idx + 1) % write_interval == 0 or (ep_idx + 1) == num_episodes:
            batch = pa.Table.from_pylist(pending, schema=SCHEMA)
            if table is None:
                table = db.create_table(table_name, batch, mode="overwrite")
            else:
                table.add(batch)
            pending = []
            print(f"    ✓ flushed {total} total steps → {table_name}.lance")

    await ctx.close()
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Server management
# ──────────────────────────────────────────────────────────────────────────────

def start_open_apps(open_apps_dir: str) -> subprocess.Popen:
    """Start OpenApps via 'uv run launch.py' in the given directory."""
    proc = subprocess.Popen(
        ["uv", "run", "launch.py"],
        cwd=open_apps_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Started OpenApps (pid {proc.pid}), waiting for startup...")
    time.sleep(8)
    return proc


def wait_for_server(base_url: str, timeout: int = 60):
    """Block until base_url returns HTTP 200 or timeout."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base_url, timeout=3)
            print(f"  Server ready at {base_url}")
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Server at {base_url} did not respond within {timeout}s")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

async def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(output_dir))

    apps_to_collect = args.apps or list(APP_ROUTES.keys())
    unknown = [a for a in apps_to_collect if a not in APP_ROUTES]
    if unknown:
        print(f"Unknown app(s): {unknown}. Available: {list(APP_ROUTES)}")
        sys.exit(1)

    server_proc = None
    if args.open_apps_dir:
        server_proc = start_open_apps(args.open_apps_dir)

    wait_for_server(args.base_url)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.no_headless)

        all_steps = 0
        all_pending: list[dict] = []   # for merged table

        for app_name in apps_to_collect:
            route      = APP_ROUTES[app_name]
            table_name = f"openapps_{app_name}"
            n = await collect_app(
                browser=browser,
                base_url=args.base_url,
                app_name=app_name,
                route=route,
                num_episodes=args.num_episodes,
                episode_len=args.episode_len,
                db=db,
                table_name=table_name,
                viewport=(args.viewport_w, args.viewport_h),
                write_interval=args.write_interval,
            )
            all_steps += n

        # Merge all per-app tables into one combined table for training
        if len(apps_to_collect) > 1:
            print("\nBuilding merged table openapps_all.lance ...")
            merged = None
            ep_offset = 0
            for app_name in apps_to_collect:
                tbl = db.open_table(f"openapps_{app_name}")
                batch = tbl.to_arrow()
                # Re-index episodes globally so they don't collide
                new_ep = pa.array(
                    [r + ep_offset for r in batch["episode_idx"].to_pylist()], type=pa.int32()
                )
                batch = batch.set_column(batch.schema.get_field_index("episode_idx"), "episode_idx", new_ep)
                max_ep = max(batch["episode_idx"].to_pylist())
                ep_offset = max_ep + 1
                if merged is None:
                    merged = db.create_table("openapps_all", batch, mode="overwrite")
                else:
                    merged.add(batch)
            print(f"  ✓ openapps_all.lance written ({all_steps} total rows)")

        await browser.close()

    if server_proc:
        server_proc.terminate()

    print(f"\nDone. Total steps collected: {all_steps}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"\nTo train on a single app:")
    print(f"  export LOCAL_DATASET_DIR={output_dir.resolve()}")
    print(f"  python train.py --config-name web_lewm data.dataset.name=openapps_todo.lance")
    print(f"\nTo train on all apps merged:")
    print(f"  python train.py --config-name web_lewm data.dataset.name=openapps_all.lance")


def parse_args():
    p = argparse.ArgumentParser(description="Collect OpenApps trajectories")
    p.add_argument("--base-url",      default="http://localhost:5001",
                   help="Base URL of the running OpenApps server")
    p.add_argument("--open-apps-dir", default=None,
                   help="If set, runs 'uv run launch.py' in this directory before recording")
    p.add_argument("--apps",          nargs="+", default=None,
                   choices=list(APP_ROUTES.keys()),
                   help="Which sub-apps to record (default: all)")
    p.add_argument("--output-dir",    default="datasets",
                   help="Directory to write LanceDB tables")
    p.add_argument("--num-episodes",  type=int, default=200)
    p.add_argument("--episode-len",   type=int, default=48)
    p.add_argument("--viewport-w",    type=int, default=1280)
    p.add_argument("--viewport-h",    type=int, default=800)
    p.add_argument("--write-interval", type=int, default=10,
                   help="Flush to disk every N episodes")
    p.add_argument("--no-headless",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
