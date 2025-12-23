import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.datamodule import VAEDataModule
from src.models import VAE
from src.callbacks import ImageLogCallback

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('medium')
    
    print(f"Starting Training: {cfg.data.dataset_name}")
    
    # 1. Data
    dm = VAEDataModule(
        dataset_name=cfg.data.dataset_name,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size
    )
    
    # 2. Model Configuration
    input_channels = 1 if cfg.data.dataset_name == "fashion_mnist" else 3

    # --- ARCHITECTURE ---
    # Convert ListConfig (from Hydra) to regular python list
    # This allows taking [32, 64] for F-MNIST or [64, 128, 256] for CIFAR from config
    hidden_dims = list(cfg.model.hidden_dims)

    # --- OPTIMIZER & SCHEDULER ---
    weight_decay = cfg.trainer.get("weight_decay", 0.0)
    betas = tuple(cfg.trainer.get("betas", [0.9, 0.999]))
    
    use_scheduler = cfg.trainer.get("use_scheduler", False)
    scheduler_patience = cfg.trainer.get("scheduler_patience", 3)
    scheduler_factor = cfg.trainer.get("scheduler_factor", 0.5)
    min_lr = cfg.trainer.get("min_lr", 1e-6)

    # --- KL ANNEALING ---
    kl_annealing = cfg.model.get("kl_annealing", False)
    kl_start = cfg.model.get("kl_start", 0.0)
    kl_end = cfg.model.get("kl_end", 1.0)
    kl_annealing_epochs = cfg.model.get("kl_annealing_epochs", 10)
    kl_warmup_epochs = cfg.model.get("kl_warmup_epochs", 0)

    # Init Model
    model = VAE(
        input_channels=input_channels,
        latent_dim=cfg.model.latent_dim,
        hidden_dims=hidden_dims,
        image_size=cfg.data.image_size,
        lr=cfg.model.lr,
        kl_annealing=kl_annealing,
        kl_start=kl_start,
        kl_end=kl_end,
        kl_annealing_epochs=kl_annealing_epochs,
        kl_warmup_epochs=kl_warmup_epochs,
        weight_decay=weight_decay,
        betas=betas,
        use_scheduler=use_scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        min_lr=min_lr
    )

    # 3. Logger & Callbacks
    logger = TensorBoardLogger(save_dir=cfg.logger.save_dir, name=f"{cfg.logger.name}_{cfg.data.dataset_name}")

    # Save checkpoints every N epochs + last one, ignoring metrics (due to VAE loss specifics)
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{logger.log_dir}/checkpoints",
        filename="vae-epoch{epoch:02d}",
        save_last=True,
        every_n_epochs=10,
        save_top_k=-1,
    )
    
    callbacks = [
        checkpoint_cb, 
        LearningRateMonitor(logging_interval='step'), 
        ImageLogCallback()
    ]
    
    # 4. Trainer
    clip_val = cfg.trainer.get("gradient_clip_val", 0.0)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=clip_val
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    train()
