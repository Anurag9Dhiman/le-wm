"""LanceDB dataset for web LeWM training."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class LanceDataset(Dataset):
    """Loads sequential (episode, step) windows from a LanceDB table.

    Each __getitem__ returns a dict mapping column name → tensor of shape
    (num_steps, *feature_dims), covering num_steps consecutive steps
    from the same episode with the given frameskip.
    """

    def __init__(self, name, cache_dir, num_steps, frameskip=1,
                 keys_to_load=None, keys_to_cache=None, transform=None, **kwargs):
        import lancedb
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.keys_to_load = keys_to_load or ["pixels", "action"]
        self.transform = transform

        db_path = Path(cache_dir)
        table_name = name.replace(".lance", "")
        db = lancedb.connect(str(db_path))
        tbl = db.open_table(table_name)

        # Load everything into memory as numpy arrays keyed by column
        self._data = {}
        df = tbl.to_pandas()

        # Build (episode_idx, step_idx) index for all valid windows
        self._index = []
        for ep_idx, group in df.groupby("episode_idx"):
            group = group.sort_values("step_idx").reset_index(drop=True)
            ep_len = len(group)
            window = (num_steps - 1) * frameskip + 1
            for start in range(ep_len - window + 1):
                self._index.append((ep_idx, start))

        # Store per-episode data for fast lookup
        self._episodes = {}
        for ep_idx, group in df.groupby("episode_idx"):
            group = group.sort_values("step_idx").reset_index(drop=True)
            ep_data = {}
            for col in self.keys_to_load:
                arr = np.stack(group[col].values)
                ep_data[col] = arr
            self._episodes[ep_idx] = ep_data

        # Cache all action data for normalizer computation
        all_actions = np.concatenate(
            [ep["action"] for ep in self._episodes.values()], axis=0
        )
        self._col_cache = {"action": all_actions}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_idx, start = self._index[idx]
        ep = self._episodes[ep_idx]
        step_indices = [start + i * self.frameskip for i in range(self.num_steps)]

        sample = {}
        for col in self.keys_to_load:
            frames = ep[col][step_indices]            # (num_steps, feat_dim)
            sample[col] = torch.from_numpy(frames).float()

        # pixels: reshape from flat to (num_steps, H, W, C) → (num_steps, C, H, W)
        if "pixels" in sample:
            t = sample["pixels"]                      # (num_steps, 150528)
            t = t.reshape(self.num_steps, 224, 224, 3)
            t = t.permute(0, 3, 1, 2)                # (num_steps, 3, H, W)
            sample["pixels"] = t

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_dim(self, col):
        """Return feature dimension of a column (used to set action_encoder.input_dim)."""
        ep = next(iter(self._episodes.values()))
        return ep[col].shape[-1]

    def get_col_data(self, col):
        """Return all data for a column (used to compute normalizer stats)."""
        return self._col_cache.get(col, np.concatenate(
            [ep[col] for ep in self._episodes.values()], axis=0
        ))
