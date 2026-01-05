import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datamodule import VAEDataModule
from src.models import VAE


def get_test_loader(cfg, device):
    dm = VAEDataModule(
        dataset_name=cfg.data.dataset_name,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
    )
    dm.setup()
    return dm.test_dataloader()


def find_worst_reconstructions(model, loader, device, k=16):
    """
    Goes through the entire dataset, calculates MSE for each image
    and returns k images with the highest error.
    """
    model.eval()
    all_losses = []
    all_originals = []
    all_recons = []

    print("Analyzing all test images to find worst reconstructions...")

    with torch.no_grad():
        for batch in tqdm(loader):
            imgs, _ = batch
            imgs = imgs.to(device)

            recons, _, _ = model(imgs)

            # Calculate MSE separately for each image in the batch
            # (B, C, H, W) -> (B, -1) -> mean(dim=1) = (B)
            mse_per_image = (
                F.mse_loss(recons, imgs, reduction="none")
                .view(imgs.size(0), -1)
                .mean(dim=1)
            )

            # Save to CPU to avoid filling GPU memory
            all_losses.append(mse_per_image.cpu())
            all_originals.append(imgs.cpu())
            all_recons.append(recons.cpu())

    # Combine everything into one large tensor
    all_losses = torch.cat(all_losses)
    all_originals = torch.cat(all_originals)
    all_recons = torch.cat(all_recons)

    # Find top-k largest errors
    # topk returns (values, indices)
    worst_values, worst_indices = torch.topk(all_losses, k)

    worst_orig = all_originals[worst_indices]
    worst_rec = all_recons[worst_indices]

    return worst_orig, worst_rec, worst_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Config & Setup Dir
    cfg = OmegaConf.load(args.config)
    experiment_name = f"{cfg.logger.name}_{cfg.data.dataset_name}"
    save_dir = os.path.join("results", experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    # Load Model
    print(f"Loading model from {args.ckpt}")
    model = VAE.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # 1. NORMAL GENERATION (Random Samples)
    print("1. Generating standard random samples...")
    with torch.no_grad():
        # z ~ N(0, 1)
        samples = model.sample(64)

    grid = torchvision.utils.make_grid(samples, nrow=8, normalize=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Standard Generated Samples (Normal Distribution)")
    plt.savefig(os.path.join(save_dir, "1_generated_standard.png"))
    plt.close()

    # 2. STANDARD RECONSTRUCTION (First Batch)
    print("2. Generating standard reconstructions...")
    loader = get_test_loader(cfg, device)
    real_img, _ = next(iter(loader))
    real_img = real_img[:8].to(device)

    with torch.no_grad():
        recons, _, _ = model(real_img)

    combo = torch.cat([real_img, recons])
    grid_rec = torchvision.utils.make_grid(combo, nrow=8, normalize=False)

    plt.figure(figsize=(12, 4))
    plt.imshow(grid_rec.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Top: Real | Bottom: Reconstructed (Random Batch)")
    plt.savefig(os.path.join(save_dir, "2_reconstructions_standard.png"))
    plt.close()

    # 3. WORST RECONSTRUCTION (Highest Error)
    print("3. Finding worst reconstructions...")
    w_orig, w_rec, w_vals = find_worst_reconstructions(model, loader, device, k=16)

    # Create grid
    # Top 2 rows: Original, Bottom 2 rows: Reconstructed
    # Or alternate: Orig, Recon, Orig, Recon...
    # Let's keep it simple: Top 2 rows = Originals, Bottom 2 rows = Recons

    # For visualization it's better to split into pairs
    # But for make_grid it's simpler to concatenate
    combo_worst = torch.cat([w_orig, w_rec])
    grid_worst = torchvision.utils.make_grid(
        combo_worst, nrow=8, normalize=False
    )  # 8 per row -> 2 rows orig, 2 rows recon

    plt.figure(figsize=(12, 6))
    plt.imshow(grid_worst.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"Failure Gallery: Worst Reconstructions (Max MSE: {w_vals[0]:.4f})")
    plt.savefig(os.path.join(save_dir, "3_worst_reconstructions.png"))
    plt.close()

    # 4. EXTREME GENERATION ("Worst" Random)
    print("4. Generating extreme samples (Outliers)...")
    # Here we take noise with larger variance (e.g., sigma=3.0)
    # These are points far from the distribution center, where the model "didn't learn"
    with torch.no_grad():
        z_extreme = torch.randn(64, model.hparams.latent_dim).to(device) * 3.0
        samples_extreme = model.decoder(z_extreme)

    grid_ext = torchvision.utils.make_grid(samples_extreme, nrow=8, normalize=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_ext.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Extreme Samples (Sigma=3.0) - Likely Artifacts")
    plt.savefig(os.path.join(save_dir, "4_generated_extreme.png"))
    plt.close()

    print(f"Done! Check results in {save_dir}/")


if __name__ == "__main__":
    main()
