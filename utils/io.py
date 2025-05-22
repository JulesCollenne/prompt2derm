import os
import json
from PIL import Image
import torchvision.transforms as T


def save_image(image, path, format="PNG"):
    """
    Save a PIL image to disk.

    Args:
        image (PIL.Image or torch.Tensor): Image to save.
        path (str): Path to save the image.
        format (str): Image format (default: PNG).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not isinstance(image, Image.Image):
        # Convert from torch.Tensor to PIL.Image
        image = tensor_to_pil(image)

    image.save(path, format=format)


def tensor_to_pil(tensor):
    """
    Converts a normalized tensor (C, H, W) in [-1, 1] to a PIL image.
    """
    inv_transform = T.Compose([
        T.Normalize(mean=[-1], std=[2]),  # undo normalization from [-1, 1] to [0, 1]
        T.Lambda(lambda x: x.clamp(0, 1)),
        T.ToPILImage()
    ])
    return inv_transform(tensor.cpu())


def load_json(path):
    """
    Loads a JSON file from disk.
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path, indent=2):
    """
    Saves a dictionary or list as a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def list_image_files(directory, extensions=('.png', '.jpg', '.jpeg')):
    """
    Returns a sorted list of image file paths in a directory.
    """
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ])
