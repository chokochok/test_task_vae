import lightning as L
import torch
import torchvision
from lightning.pytorch.loggers import TensorBoardLogger


class ImageLogCallback(L.Callback):
    def __init__(self, num_samples=8, log_every_n_epochs=1):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        images, _ = batch
        images = images.to(pl_module.device)[: self.num_samples]

        pl_module.eval()
        with torch.no_grad():
            recons, _, _ = pl_module(images)
            random_samples = pl_module.sample(self.num_samples)
        pl_module.train()

        # Grid: Top=Original, Bottom=Recon
        grid_recon = torchvision.utils.make_grid(
            torch.cat([images, recons]), nrow=self.num_samples, normalize=False
        )
        grid_samples = torchvision.utils.make_grid(
            random_samples, nrow=4, normalize=False
        )

        if isinstance(trainer.logger, TensorBoardLogger):
            tensorboard = trainer.logger.experiment
            tensorboard.add_image("Reconstruction", grid_recon, trainer.global_step)
            tensorboard.add_image("Random Samples", grid_samples, trainer.global_step)
