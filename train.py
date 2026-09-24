import json
import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


CHECKPOINT_INTERVAL = 500


class ResumableDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_idx = 0

    def state_dict(self):
        return {"start_idx": self._start_idx}

    def load_state_dict(self, state):
        self._start_idx = state.get("start_idx", 0)

    def __iter__(self):
        iterator = super().__iter__()
        for i, batch in enumerate(iterator):
            if i < self._start_idx:
                continue
            self._start_idx = 0
            yield batch
        self._start_idx = 0


class RollingModelCheckpoint(ModelCheckpoint):
    def __init__(self, status_path, **kwargs):
        self.status_path = Path(status_path)
        super().__init__(**kwargs)

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)

        if not trainer.is_global_zero:
            return

        checkpoint_path = Path(filepath)
        if not checkpoint_path.is_file():
            print(f"WARNING: checkpoint does not exist: {checkpoint_path}")
            return

        metadata = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "global_step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "file_size_bytes": checkpoint_path.stat().st_size,
        }

        tmp_path = self.status_path.with_suffix(
            self.status_path.suffix + ".tmp"
        )

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, self.status_path)
            print(
                f"[Checkpoint] Saved {checkpoint_path.name} "
                f"at global_step={trainer.global_step}, "
                f"epoch={trainer.current_epoch}"
            )
        except Exception as exc:
            print(f"WARNING: failed to write checkpoint metadata: {exc}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass


def lejepa_forward1(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]

    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    self.log_dict(
        {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k},
        on_step=True,
        sync_dist=True,
    )

    return output


def lejepa_forward(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    eqm_weight = cfg.loss.get("eqm_pred_weight", 0.5)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]

    pred_emb_std = self.model.predict(ctx_emb, ctx_act)
    output["pred_loss"] = (pred_emb_std - tgt_emb).pow(2).mean()

    ctx_actions_raw = batch["action"][:, :ctx_len]
    B = ctx_actions_raw.shape[0]

    with torch.enable_grad():
        gamma = torch.rand(
            B, 1, 1,
            device=ctx_actions_raw.device,
            dtype=ctx_actions_raw.dtype,
        )
        eps = torch.randn_like(ctx_actions_raw)
        eps1 = torch.randn_like(ctx_emb)

        act_gamma = (
            gamma * ctx_actions_raw.detach()
            + (1.0 - gamma) * eps
        ).requires_grad_(True)

        ctx_n_emb = (
            gamma * ctx_emb.detach()
            + (1.0 - gamma) * eps1
        )

        pred_emb_noisy = self.model.predict(
            ctx_n_emb,
            self.model.action_encoder(act_gamma),
        )

        energy = (
            pred_emb_noisy - tgt_emb.detach()
        ).pow(2).mean()

        grad_energy = torch.autograd.grad(
            energy,
            act_gamma,
            create_graph=True,
        )[0]

        target_grad = eps - ctx_actions_raw.detach()

    grad_energy = F.normalize(
        grad_energy.flatten(1),
        dim=1,
        eps=1e-8,
    )

    target_grad = F.normalize(
        target_grad.flatten(1),
        dim=1,
        eps=1e-8,
    )

    cosine_loss = 1.0 - (grad_energy * target_grad).sum(dim=1)
    weight = (1.0 - gamma).flatten()

    output["pred_loss_eqm"] = (
        weight * cosine_loss
    ).sum() / weight.sum().clamp_min(1e-8)

    output["energy"] = energy.detach()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    output["loss"] = (
        output["pred_loss"]
        + eqm_weight * output["pred_loss_eqm"]
        + lambd * output["sigreg_loss"]
    )

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    losses_dict[f"{stage}/energy"] = output["energy"]

    self.log_dict(
        losses_dict,
        on_step=True,
        sync_dist=True,
    )

    return output


def get_latest_checkpoint(run_dir: Path, model_name: str):
    checkpoint_path = run_dir / f"{model_name}_last.ckpt"
    status_path = run_dir / "checkpoint_status.json"

    if not checkpoint_path.is_file():
        print("No rolling checkpoint found. Starting a fresh run.")
        return None

    if status_path.is_file():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)

            print("\n" + "=" * 72)
            print("RESUMING FROM ROLLING CHECKPOINT")
            print("=" * 72)
            print(f"Checkpoint : {status.get('checkpoint')}")
            print(f"Global step: {status.get('global_step')}")
            print(f"Epoch      : {status.get('epoch')}")
            print(f"Saved at   : {status.get('saved_at')}")
            print(f"File size  : {status.get('file_size_bytes')} bytes")
            print("=" * 72 + "\n")
        except Exception as exc:
            print(f"WARNING: could not read {status_path}: {exc}")
    else:
        print(f"Found checkpoint but no metadata file: {checkpoint_path}")

    return checkpoint_path


def remove_old_step_checkpoints(run_dir: Path, model_name: str):
    old_checkpoints = list(run_dir.glob(f"{model_name}_step*.ckpt"))

    if not old_checkpoints:
        return

    print(f"Removing {len(old_checkpoints)} old step-based checkpoint(s)...")

    for checkpoint in old_checkpoints:
        try:
            checkpoint.unlink()
            print(f"Removed: {checkpoint.name}")
        except OSError as exc:
            print(f"Could not remove {checkpoint}: {exc}")


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="lewm",
)
def run(cfg):
    dataset = swm.data.HDF5Dataset(
        **cfg.data.dataset,
        transform=None,
    )

    transforms = [
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            transforms.append(
                get_column_normalizer(dataset, col, col)
            )

            setattr(
                cfg.wm,
                f"{col}_dim",
                dataset.get_dim(col),
            )

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = ResumableDataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )

    val = ResumableDataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)

    effective_act_dim = (
        cfg.data.dataset.frameskip * cfg.wm.action_dim
    )

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(
        input_dim=effective_act_dim,
        emb_dim=embed_dim,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    steps_per_epoch = len(train)
    total_steps = cfg.trainer.max_epochs * steps_per_epoch
    warmup_steps = int(0.03 * total_steps)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": total_steps,
            },
            "interval": "step",
        },
    }

    data_module = spt.data.DataModule(
        train=train,
        val=val,
    )

    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_dir = Path(
        "/kaggle/working",
        cfg.get("subdir") or "lewm_run",
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    remove_old_step_checkpoints(
        run_dir,
        cfg.output_model_name,
    )

    logger = None

    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(
            OmegaConf.to_container(cfg, resolve=True)
        )

    checkpoint_path = (
        run_dir / f"{cfg.output_model_name}_last.ckpt"
    )
    status_path = run_dir / "checkpoint_status.json"

    step_checkpoint = RollingModelCheckpoint(
        status_path=status_path,
        dirpath=run_dir,
        filename=f"{cfg.output_model_name}_last",
        every_n_train_steps=CHECKPOINT_INTERVAL,
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
    )

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            step_checkpoint,
            object_dump_callback,
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    latest_ckpt = get_latest_checkpoint(
        run_dir,
        cfg.output_model_name,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=latest_ckpt,
    )

    manager()


if __name__ == "__main__":
    run()
