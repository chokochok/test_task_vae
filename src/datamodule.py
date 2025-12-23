import lightning as L
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image

class SimpleCelebADataset(Dataset):
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
        return image, 0 

class VAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        image_size: int
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transform = self._get_transforms()

    def _get_transforms(self):
        transform_list = []
        
        if self.hparams.dataset_name == "celeba":
            # Crop 178x178 (face) -> Resize
            transform_list.append(transforms.CenterCrop(178))
            transform_list.append(transforms.Resize((self.hparams.image_size, self.hparams.image_size)))
        else:
            # Resize for CIFAR/Fashion
            transform_list.append(transforms.Resize((self.hparams.image_size, self.hparams.image_size)))
            
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    def prepare_data(self):
        name = self.hparams.dataset_name
        d_dir = self.hparams.data_dir
        
        if name == "cifar10":
            datasets.CIFAR10(d_dir, train=True, download=True)
            datasets.CIFAR10(d_dir, train=False, download=True)
        elif name == "fashion_mnist":
            datasets.FashionMNIST(d_dir, train=True, download=True)
            datasets.FashionMNIST(d_dir, train=False, download=True)
        elif name == "celeba":
            celeba_path = Path(d_dir) / "celeba" / "img_align_celeba"
            if not celeba_path.exists():
                 print(f"WARNING: CelebA not found. Please place images in {celeba_path}")

    def setup(self, stage=None):
        name = self.hparams.dataset_name
        d_dir = self.hparams.data_dir
        
        if name == "cifar10":
            full = datasets.CIFAR10(d_dir, train=True, transform=self.transform)
            test = datasets.CIFAR10(d_dir, train=False, transform=self.transform)
        elif name == "fashion_mnist":
            full = datasets.FashionMNIST(d_dir, train=True, transform=self.transform)
            test = datasets.FashionMNIST(d_dir, train=False, transform=self.transform)
        elif name == "celeba":
            img_path = Path(d_dir) / "celeba" / "img_align_celeba"
            full_data = SimpleCelebADataset(img_path, transform=self.transform)
            
            total = len(full_data)
            train_sz = int(0.85 * total)
            val_sz = int(0.05 * total)
            test_sz = total - train_sz - val_sz
            self.train_ds, self.val_ds, self.test_ds = random_split(full_data, [train_sz, val_sz, test_sz])
            return

        # Common split
        if name != "celeba":
            train_len = int(0.9 * len(full))
            val_len = len(full) - train_len
            self.train_ds, self.val_ds = random_split(full, [train_len, val_len])
            self.test_ds = test

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
