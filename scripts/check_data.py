import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import hydra
import matplotlib.pyplot as plt
import torchvision
from omegaconf import DictConfig, OmegaConf

from src.datamodule import VAEDataModule


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def check_data(cfg: DictConfig):
    print(f"Loading configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Init DataModule
    dm = VAEDataModule(
        dataset_name=cfg.data.dataset_name,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
    )

    print("Preparing data (downloading if needed)...")
    dm.prepare_data()

    print("Setting up data splits...")
    dm.setup()

    # Get a batch
    loader = dm.train_dataloader()
    images, labels = next(iter(loader))

    print(f"Batch shape: {images.shape}")
    print(f"Data range: Min={images.min().item():.3f}, Max={images.max().item():.3f}")

    # Check if channels match config
    expected_channels = 3 if cfg.data.dataset_name != "fashion_mnist" else 1
    assert images.shape[1] == expected_channels, (
        f"Expected {expected_channels} channels, got {images.shape[1]}"
    )

    # Create visualization
    grid = torchvision.utils.make_grid(images[:16], nrow=4, normalize=False)

    plt.figure(figsize=(10, 10))
    # Permute from (C, H, W) to (H, W, C) for matplotlib
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Sample Batch: {cfg.data.dataset_name}")

    output_path = f"data/preview/data_preview_{cfg.data.dataset_name}.png"
    plt.savefig(output_path)
    print(f"Saved data preview to {output_path}")
    print("Data check passed successfully!")


if __name__ == "__main__":
    check_data()
