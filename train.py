import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg, VISReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback
from data.lance_dataset import LanceDataset

# UntypedStorage._share_filename_cpu_ is called when spawn workers try to
# share tensors via file-backed memory. This patch is a no-op on CUDA/CPU
# but prevents errors if non-CPU tensors end up in the pickle chain.
def _patch_storage_sharing():
    import torch.storage as _ts
    _orig = _ts.UntypedStorage._share_filename_cpu_
    def _safe(self, *args, **kwargs):
        if self.device.type not in ('cpu', 'cuda'):
            return self.cpu()._share_filename_cpu_(*args, **kwargs)
        return _orig(self, *args, **kwargs)
    _ts.UntypedStorage._share_filename_cpu_ = _safe

_patch_storage_sharing()


# spt.data.Subset stores the full Lightning trainer on self._trainer via
# set_pl_trainer, but has no __getstate__, so the trainer gets pickled when
# spawn workers copy the dataset. Exclude _trainer from pickle state to avoid
# dragging the entire model into each worker process.
def _patch_subset_pickling():
    from stable_pretraining.data.datasets import Subset as _Subset
    def _getstate(self):
        state = self.__dict__.copy()
        state['_trainer'] = None
        return state
    _Subset.__getstate__ = _getstate

_patch_subset_pickling()


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd_sig = cfg.loss.sigreg.weight
    lambd_vis = (cfg.loss.visreg.weight
                 if hasattr(cfg.loss, "visreg") else 0.0)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    emb_t = emb.transpose(0, 1)  # (T, B, D)
    output["pred_loss"]   = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb_t)
    output["visreg_loss"] = self.visreg(emb_t)
    output["loss"] = (output["pred_loss"]
                      + lambd_sig * output["sigreg_loss"]
                      + lambd_vis * output["visreg_loss"])  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    if dataset_name.endswith(".lance"):
        dataset = LanceDataset(
            name=dataset_name, cache_dir=cache_dir, transform=None, **dataset_cfg
        )
    else:
        dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    import math
    steps_per_epoch = math.ceil(len(train_set) / cfg.loader.batch_size)
    max_steps = steps_per_epoch * cfg.trainer.max_epochs
    warmup_steps = max(1, int(0.05 * max_steps))

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
            },
            "interval": "step",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    vis_kwargs = (cfg.loss.visreg.kwargs
                  if hasattr(cfg.loss, "visreg") else {"num_projections": 256})
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        visreg = VISReg(**vis_kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir()) / 'checkpoints' / run_id

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
