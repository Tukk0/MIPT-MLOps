"""
Training entry with Hydra + PyTorch Lightning.

Usage:
    python train.py
    python train.py training.max_epochs=50 training.optimizer.lr=0.0005
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import git
import hydra
from fire import Fire
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

warnings.filterwarnings(
    "default",
    message=".*requires PyTorch.*",
    category=UserWarning,
    module="torchvision",
)


def _get_git_commit() -> str:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        return "unknown"


def train(cfg: DictConfig) -> None:
    """Training function for fire CLI."""
    git_sha = _get_git_commit()
    print(f"Training started — git commit: {git_sha}")

    log_dir = str(Path.cwd() / "logs" / f"run_{int(time.time())}")
    mlflow_logger = MLFlowLogger(
        experiment_name="digits",
        tracking_uri=cfg.logging.mlflow_uri,
    )
    mlflow_logger.log_params(
        {
            "optimizer_name": cfg.training.optimizer.name,
            "optimizer_lr": cfg.training.optimizer.lr,
            "optimizer_weight_decay": cfg.training.optimizer.weight_decay,
            "scheduler_name": cfg.training.scheduler.name,
            "max_epochs": cfg.training.max_epochs,
            "gradient_clip_val": cfg.training.gradient_clip_val,
            "devices": str(cfg.training.devices),
            "batch_size": cfg.data.batch_size,
            "num_workers": cfg.data.num_workers,
            "git_commit": git_sha,
        }
    )

    tb_logger = TensorBoardLogger(log_dir, name="digits")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.early_stopping.patience,
        mode="min",
        verbose=True,
    )
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.2f}",
    )

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        devices=cfg.training.devices,
        logger=[mlflow_logger, tb_logger],
        callbacks=[early_stop, checkpoint_cb],
        gradient_clip_val=cfg.training.gradient_clip_val,
        precision=cfg.training.precision,
        default_root_dir=log_dir,
    )

    from src.data_module import MNISTDataModule
    from src.model import DigitModel

    data_module = MNISTDataModule(
        data_dir="./data",
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=tuple(cfg.data.image_size),
        mean=cfg.data.mean,
        std=cfg.data.std,
    )

    model = DigitModel(
        name=cfg.model.name,
        optimizer_cfg=cfg.training.optimizer,
        scheduler_cfg=cfg.training.scheduler,
        conv1_out=cfg.model.layers[0].out_channels,
        conv2_out=cfg.model.layers[1].out_channels,
        fc1_features=cfg.model.layers[2].out_features,
        fc2_features=cfg.model.layers[3].out_features,
        dropout_prob=cfg.model.dropout_prob,
        batch_norm=cfg.model.batch_norm,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    print(f"Training complete. Logs saved to: {log_dir}")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    Fire(main)
