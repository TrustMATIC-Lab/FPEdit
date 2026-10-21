
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LatentDataset(Dataset):
    def __init__(self, path: Path, num_ws: int):
        self.latents = torch.load(path, map_location="cpu")
        if not torch.is_tensor(self.latents) or self.latents.ndim != 3:
            raise ValueError(f"{path} must contain a W+ tensor")
        if tuple(self.latents.shape[1:]) != (num_ws, 512):
            raise ValueError(
                f"Expected latent shape (N,{num_ws},512), got {tuple(self.latents.shape)}"
            )

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):
        return self.latents[index]


class ImageDataset(Dataset):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, root: Path):
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in self.extensions
        )
        if not self.paths:
            raise ValueError(f"No images found in {root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name
