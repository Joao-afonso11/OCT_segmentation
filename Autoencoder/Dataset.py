# datasets/oct_mae_dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class OCTMaskedDataset(Dataset):
    def __init__(self, root_dir, patch_size=16, masking_ratio=0.4):
        super().__init__()
        self.root_dir = root_dir
        self.filenames = sorted(os.listdir(root_dir))
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size
        self.num_patches = (256 // patch_size) ** 2

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(path)
        image = self.transform(image)

        # Generate random binary mask: True where patch is masked
        mask = torch.zeros(self.num_patches, dtype=torch.bool)
        num_mask = int(self.masking_ratio * self.num_patches)
        mask_indices = random.sample(range(self.num_patches), num_mask)
        mask[mask_indices] = True

        return image, mask
