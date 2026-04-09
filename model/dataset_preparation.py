# This file does a few more steps of preprocessing on the
# images and masks when they are called upon and used for training.

# ============================================================
#                       Import Statements
# ============================================================
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ============================================================
#                       Class 
# ============================================================
class houseDataset(Dataset):
    def __init__(self, split, image_size=256):
        self.image_dir = f"data/images/{split}"
        self.mask_dir = f"data/masks/{split}"
        self.image_size = image_size

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filesnames = sorted(os.listdir(self.mask_dir))

        # Image transformation
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Mask transformation 
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        mask_filename = self.mask_filesnames[index]

        image = Image.open(os.path.join(self.image_dir, image_filename)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_filename)).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return image, mask

    def __len__(self):
        return len(self.image_filenames)

# Testing code
if __name__ == "__main__":
    dataset = houseDataset("train")
    print(f"Number of images: {len(dataset)}")
    
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image min/max: {image.min():.2f} / {image.max():.2f}")
    print(f"Mask unique values: {mask.unique()}")