import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json

class SkinLesionDataset(Dataset):
    """
    Dataset class to load dermoscopic images and labels from a given directory.
    Supports real and synthetic datasets with optional prompt annotations.
    """
    def __init__(self, image_dir, label_file=None, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Optional: load label file (assumes {filename: label} format)
        self.labels = None
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.labels = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        if self.labels:
            label = self.labels.get(os.path.basename(path), 0)
        else:
            label = 0  # default to benign

        return image, label

def get_dataloader(image_dir, label_file=None, batch_size=32, shuffle=True, num_workers=4):
    dataset = SkinLesionDataset(image_dir=image_dir, label_file=label_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
