import lightning as L
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from pathlib import Path
import zipfile
from PIL import Image


class SimpleCelebADataset(Dataset):
    """Simple CelebA dataset for manually extracted images."""
    def __init__(self, img_folder, transform=None):
        self.img_folder = Path(img_folder)
        self.image_files = sorted(list(self.img_folder.glob("*.jpg")))
        self.transform = transform
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {img_folder}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and dummy label (CelebA doesn't need labels for VAE)
        return image, 0


class VAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "cifar10",
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 64,  # Default 64 for CelebA/General VAE
    ):
        """
        Lightning DataModule for VAE training.
        
        Args:
            dataset_name: 'cifar10', 'celeba', or 'fashion_mnist'
            data_dir: Path to download/store data
            batch_size: Batch size for loaders
            num_workers: Number of subprocesses for data loading
            image_size: Target image height/width
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Transformations will be defined in setup or property
        self.transform = self._get_transforms()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_transforms(self):
        """
        Define transformations based on the dataset requirements.
        """
        transform_list = []

        if self.dataset_name == "celeba":
            # Requirement: CelebA (crop/resize to 64x64)
            # CenterCrop 178 is standard for CelebA to focus on the face
            transform_list.append(transforms.CenterCrop(178))
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        elif self.dataset_name == "cifar10":
            # CIFAR is naturally 32x32. We can resize if needed, or keep native.
            # If image_size is explicitly set to something other than 32, we resize.
            if self.image_size != 32:
                transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        elif self.dataset_name == "fashion_mnist":
            # FashionMNIST is 28x28 grayscale.
            # Usually padded to 32 or resized.
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
            # VAE usually expects tensor [0, 1]
            transform_list.append(transforms.ToTensor())
            return transforms.Compose(transform_list)

        # Common transforms
        transform_list.append(transforms.ToTensor())
        # Note: We are NOT using Normalize((0.5,), (0.5,)) here because 
        # standard VAEs often use Sigmoid output (pixel values [0, 1]).
        # If using Tanh output in generator, we would need normalization.
        
        return transforms.Compose(transform_list)

    def prepare_data(self):
        """
        Download data if needed. This method is called only from a single GPU.
        """
        if self.dataset_name == "cifar10":
            datasets.CIFAR10(self.data_dir, train=True, download=True)
            datasets.CIFAR10(self.data_dir, train=False, download=True)
        
        elif self.dataset_name == "fashion_mnist":
            datasets.FashionMNIST(self.data_dir, train=True, download=True)
            datasets.FashionMNIST(self.data_dir, train=False, download=True)
            
        elif self.dataset_name == "celeba":
            # Manual CelebA extraction from zip file
            zip_path = Path(self.data_dir) / "img_align_celeba.zip"
            celeba_dir = Path(self.data_dir) / "celeba"
            img_folder = celeba_dir / "img_align_celeba"
            
            # Check if data is already extracted
            if img_folder.exists() and len(list(img_folder.glob("*.jpg"))) > 0:
                print(f"CelebA data already exists at {img_folder}")
                return
            
            # If zip exists, extract it
            if zip_path.exists():
                print(f"Found {zip_path}, extracting...")
                celeba_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(celeba_dir)
                print(f"Extracted CelebA to {celeba_dir}")
            else:
                raise FileNotFoundError(
                    f"CelebA zip file not found at {zip_path}.\n"
                    f"Please download from: https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM\n"
                    f"Save as: {zip_path}"
                )

    def setup(self, stage=None):
        """
        Load data, split, and wrap in Dataset objects.
        """
        if self.dataset_name == "cifar10":
            full_dataset = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
            # Split train into train/val (e.g., 45k train, 5k val)
            train_len = int(0.9 * len(full_dataset))
            val_len = len(full_dataset) - train_len
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, val_len]
            )
            self.test_dataset = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

        elif self.dataset_name == "fashion_mnist":
            full_dataset = datasets.FashionMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            train_len = int(0.9 * len(full_dataset))
            val_len = len(full_dataset) - train_len
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, val_len]
            )
            self.test_dataset = datasets.FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )

        elif self.dataset_name == "celeba":
            # Use manually extracted images with SimpleCelebADataset
            celeba_dir = Path(self.data_dir) / "celeba"
            img_folder = celeba_dir / "img_align_celeba"
            
            full_dataset = SimpleCelebADataset(img_folder, transform=self.transform)
            
            # Standard CelebA splits: 162770 train, 19867 val, 19962 test
            total = len(full_dataset)
            train_size = int(0.8 * total)  # ~130k
            val_size = int(0.1 * total)    # ~20k
            test_size = total - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    @property
    def num_channels(self):
        """Return number of channels for the model architecture."""
        if self.dataset_name == "fashion_mnist":
            return 1
        return 3
