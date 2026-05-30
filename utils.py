import json
import os
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

class _ReshapePixels:
    """Reshape flat uint8 pixels [T, C*H*W] → [T, C, H, W] before image transforms."""
    def __init__(self, source: str, target: str, c: int = 3, h: int = 224, w: int = 224):
        self.source, self.target, self.c, self.h, self.w = source, target, c, h, w

    def __call__(self, x):
        px = x[self.source]
        if px.ndim == 2:  # [T, C*H*W]
            px = px.reshape(px.shape[0], self.c, self.h, self.w)
        x[self.target] = px
        return x


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    reshape = _ReshapePixels(source=source, target=target, c=3, h=img_size, w=img_size)
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    return dt.transforms.Compose(reshape, to_image)


class ZScoreNormalizer:
    """Picklable z-score normalizer — stores mean/std as numpy arrays so it
    survives multiprocessing pickle when passed to DataLoader workers."""

    def __init__(self, mean, std):
        self.mean = mean.cpu().numpy()
        self.std = std.cpu().numpy()

    def __call__(self, x):
        import numpy as np
        m = torch.from_numpy(self.mean)
        s = torch.from_numpy(self.std)
        return ((x - m) / s).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Saves weights_latest.pt every epoch and weights_best.pt when val pred_loss improves."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            val_loss = trainer.callback_metrics.get("validate/pred_loss_epoch", None)
            val_loss = float(val_loss) if val_loss is not None else None

            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1, val_loss)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1, val_loss)

    def _save(self, model, epoch, val_loss=None):
        from omegaconf import OmegaConf
        save_dir = Path(os.environ.get("SCRATCH", ".")) / "checkpoints" / self.run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Always overwrite latest
        torch.save(model.state_dict(), save_dir / "weights_latest.pt")

        # Overwrite best only if val loss improved
        best_loss_file = save_dir / "best_loss.txt"
        best_loss = float("inf")
        if best_loss_file.exists():
            best_loss = float(best_loss_file.read_text())
        if val_loss is not None and val_loss < best_loss:
            torch.save(model.state_dict(), save_dir / "weights_best.pt")
            best_loss_file.write_text(str(val_loss))
            print(f"  New best checkpoint at epoch {epoch}: val_loss={val_loss:.4f}")

        cfg_path = save_dir / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(json.dumps(OmegaConf.to_container(self.cfg, resolve=True), indent=2))