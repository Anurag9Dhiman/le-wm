"""Merge multiple LanceDB tables into one, re-indexing episodes.

Usage:
    python data/merge_datasets.py \\
        --dataset-dir datasets \\
        --tables ai_alchemy_train openapps_todo openapps_calendar \\
        --output web_train_all
"""

import argparse
import sys
from pathlib import Path

import pyarrow as pa

try:
    import lancedb
except ImportError:
    print("pip install lancedb"); sys.exit(1)


def merge(dataset_dir: str, table_names: list[str], output_name: str):
    db  = lancedb.connect(dataset_dir)
    out = None
    ep_offset = 0

    for name in table_names:
        print(f"  reading {name}.lance ...", end="", flush=True)
        tbl   = db.open_table(name)
        batch = tbl.to_arrow()
        n     = len(batch)

        # shift episode_idx so episodes don't collide across tables
        old_ep = batch["episode_idx"].to_pylist()
        new_ep = pa.array([e + ep_offset for e in old_ep], type=pa.int32())
        batch  = batch.set_column(
            batch.schema.get_field_index("episode_idx"), "episode_idx", new_ep
        )
        ep_offset = max(new_ep.to_pylist()) + 1

        if out is None:
            out = db.create_table(output_name, batch, mode="overwrite")
        else:
            out.add(batch)
        print(f" {n:,} rows (ep_offset now {ep_offset})")

    print(f"\n✓ {output_name}.lance  total rows: {len(out)}")
    print(f"  directory: {Path(dataset_dir).resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="datasets")
    p.add_argument("--tables",      nargs="+", required=True)
    p.add_argument("--output",      default="web_train_all")
    a = p.parse_args()
    merge(a.dataset_dir, a.tables, a.output)
