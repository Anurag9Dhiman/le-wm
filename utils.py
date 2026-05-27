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
    """Picklable z-score normalizer — stores mean/std as numpy to survive
    multiprocessing pickle on MPS devices (tensor sharing is CPU-only)."""

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
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )