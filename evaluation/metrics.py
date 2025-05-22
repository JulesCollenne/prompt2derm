import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

try:
    import clip
    from PIL import Image
except ImportError:
    clip = None  # Optional: used for CLIP-based scoring


def compute_metrics(preds, labels, average="binary"):
    """
    Compute standard classification metrics.

    Args:
        preds (List[int]): predicted class indices
        labels (List[int]): ground-truth class indices
        average (str): 'binary' for 2-class, 'macro' for multi-class

    Returns:
        dict: dictionary with accuracy, precision, recall, f1
    """
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average=average, zero_division=0),
        "recall": recall_score(labels, preds, average=average, zero_division=0),
        "f1": f1_score(labels, preds, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }

    try:
        metrics["roc_auc"] = roc_auc_score(labels, preds)
    except:
        metrics["roc_auc"] = None  # e.g., invalid for >2 classes or uniform predictions

    return metrics


def compute_clip_similarity(image: Image.Image, prompt: str, model=None, preprocess=None, device="cuda"):
    """
    Compute CLIP similarity between an image and a prompt.

    Args:
        image (PIL.Image): Image to compare
        prompt (str): Text prompt
        model: Preloaded CLIP model
        preprocess: CLIP preprocessing transform
        device: torch device

    Returns:
        float: cosine similarity between image and text embeddings
    """
    if clip is None:
        raise ImportError("CLIP not installed. Run `pip install git+https://github.com/openai/CLIP.git`.")

    if model is None or preprocess is None:
        model, preprocess = clip.load("ViT-B/32", device=device)

    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    return similarity


def compute_fid(real_images_dir, fake_images_dir):
    """
    Placeholder for computing FID score between two folders of images.
    You can implement this using:
    - torch-fidelity
    - pytorch-fid
    - clean-fid

    Returns:
        float: FID score
    """
    raise NotImplementedError("FID computation requires external packages like 'clean-fid' or 'pytorch-fid'.")
