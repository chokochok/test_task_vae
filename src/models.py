import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
import lpips

# ==========================================
# 1. RESNET BLOCKS
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.silu(self.conv_block(x) + self.shortcut(x))

# ==========================================
# 2. ENCODERS
# ==========================================

class StandardEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim, hidden_dims, image_size):
        super().__init__()
        modules = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.feature_map_size = image_size // (2 ** len(hidden_dims))
        if self.feature_map_size < 1:
            self.feature_map_size = 1
        self.flatten_dim = hidden_dims[-1] * (self.feature_map_size ** 2)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_mu(x), self.fc_var(x)

class ResNetEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim, hidden_dims, image_size):
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.SiLU()
            )
        )
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(ResidualBlock(in_channels, h_dim, stride=2))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.feature_map_size = image_size // (2 ** len(hidden_dims))
        if self.feature_map_size < 1:
            self.feature_map_size = 1
        self.flatten_dim = hidden_dims[-1] * (self.feature_map_size ** 2)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_mu(x), self.fc_var(x)

# ==========================================
# 3. DECODERS
# ==========================================

class StandardDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, hidden_dims, image_size):
        super().__init__()
        self.hidden_dims = hidden_dims[::-1]
        self.image_size = image_size
        self.feature_map_size = image_size // (2 ** len(hidden_dims))
        if self.feature_map_size < 1:
            self.feature_map_size = 1
        self.flatten_dim = self.hidden_dims[0] * (self.feature_map_size ** 2)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        modules = []
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], output_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0], self.feature_map_size, self.feature_map_size)
        x = self.decoder(x)
        x = self.final_layer(x)
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
        return x

class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, hidden_dims, image_size):
        super().__init__()
        self.hidden_dims = hidden_dims[::-1]
        self.image_size = image_size
        self.feature_map_size = image_size // (2 ** len(hidden_dims))
        if self.feature_map_size < 1:
            self.feature_map_size = 1
        self.flatten_dim = self.hidden_dims[0] * (self.feature_map_size ** 2)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        modules = []
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ResidualBlock(self.hidden_dims[i], self.hidden_dims[i+1], stride=1)
                )
            )
        modules.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ResidualBlock(self.hidden_dims[-1], self.hidden_dims[-1], stride=1)
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Conv2d(self.hidden_dims[-1], output_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0], self.feature_map_size, self.feature_map_size)
        x = self.decoder(x)
        x = self.final_layer(x)
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
        return x

# ==========================================
# 4. MAIN VAE CLASS
# ==========================================

class VAE(L.LightningModule):
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        hidden_dims: list = None,
        image_size: int = 32,
        lr: float = 1e-3,
        
        loss_type: str = "MSE", 
        use_resnet: bool = False,
        perceptual_weight: float = 0.0,
        
        kl_annealing: bool = False,
        kl_start: float = 0.0,
        kl_end: float = 1.0,
        kl_annealing_epochs: int = 10,
        kl_warmup_epochs: int = 0,
        weight_decay: float = 0.0,
        betas: tuple = (0.9, 0.999),
        use_scheduler: bool = False,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        # --- ARCHITECTURE ---
        if use_resnet:
            print(">>> Using ResNet Architecture")
            self.encoder = ResNetEncoder(input_channels, latent_dim, hidden_dims, image_size)
            self.decoder = ResNetDecoder(latent_dim, input_channels, hidden_dims, image_size)
        else:
            print(">>> Using Standard Architecture")
            self.encoder = StandardEncoder(input_channels, latent_dim, hidden_dims, image_size)
            self.decoder = StandardDecoder(latent_dim, input_channels, hidden_dims, image_size)

        # --- PERCEPTUAL LOSS SETUP ---
        if perceptual_weight > 0:
            print(f">>> LPIPS (VGG) Perceptual Loss Enabled (Weight: {perceptual_weight})")
            self.lpips = lpips.LPIPS(net='vgg').eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
        else:
            self.lpips = None

        self.current_beta = kl_end if not kl_annealing else kl_start

    def forward(self, x):
        mu, log_var = self.encoder(x)
        log_var = torch.clamp(log_var, min=-10, max=10)
        z = self.reparameterize(mu, log_var)
        recons = self.decoder(z)
        return recons, mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def get_current_beta(self):
        if not self.hparams.kl_annealing:
            return self.hparams.kl_end
        
        epoch = self.current_epoch
        warmup = self.hparams.kl_warmup_epochs
        annealing_len = self.hparams.kl_annealing_epochs
        start = self.hparams.kl_start
        end = self.hparams.kl_end
        
        if epoch < warmup:
            return start
        if epoch >= (warmup + annealing_len):
            return end
        
        steps_in_annealing = epoch - warmup
        slope = (end - start) / annealing_len
        return start + slope * steps_in_annealing

    def loss_function(self, recons, input_img, mu, log_var, beta=1.0):
        loss_type = self.hparams.loss_type.upper()
        
        # 1. Pixel-wise Reconstruction (L1/MSE)
        if loss_type == "L1":
            pixel_loss = F.l1_loss(recons, input_img, reduction='sum')
        else:
            pixel_loss = F.mse_loss(recons, input_img, reduction='sum')
        
        # 2. Perceptual Loss (VGG)
        p_loss = torch.tensor(0.0, device=self.device)
        if self.lpips is not None:
            # LPIPS очікує вхід у діапазоні [-1, 1]
            # Наш VAE (Sigmoid) видає [0, 1].
            # Перетворення: x_norm = x * 2 - 1
            rec_norm = recons * 2 - 1
            in_norm = input_img * 2 - 1
            
            # lpips.forward повертає тензор (batch, 1, 1, 1), тому беремо sum
            lpips_val = self.lpips(rec_norm, in_norm)
            p_loss = torch.sum(lpips_val)
            
            # Додаємо до загального лоссу з вагою
            pixel_loss = pixel_loss + (self.hparams.perceptual_weight * p_loss)

        # 3. KL Divergence
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total
        loss = pixel_loss + beta * kld_loss
        
        batch_size = input_img.size(0)
        # Повертаємо нормалізовані значення для логування
        return loss / batch_size, pixel_loss / batch_size, kld_loss / batch_size

    def training_step(self, batch, batch_idx):
        real_img, _ = batch
        recons, mu, log_var = self(real_img)
        
        self.current_beta = self.get_current_beta()
        loss, recons_loss, kld_loss = self.loss_function(recons, real_img, mu, log_var, beta=self.current_beta)
        
        self.log_dict({
            "train_loss": loss,
            "train_recon": recons_loss,
            "train_kld": kld_loss,
            "beta": self.current_beta
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        real_img, _ = batch
        recons, mu, log_var = self(real_img)
        loss, recons_loss, kld_loss = self.loss_function(recons, real_img, mu, log_var, beta=self.current_beta)
        
        self.log_dict({
            "val_loss": loss,
            "val_recon": recons_loss,
            "val_kld": kld_loss
        }, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas
        )
        if self.hparams.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience, min_lr=self.hparams.min_lr, verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}}
        return optimizer

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)
        return self.decoder(z)
