import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier.model import get_model
from dataset.loader import SkinLesionDataset
from evaluation.metrics import compute_metrics
from utils.io import load_json

def evaluate_model(
    model_path,
    data_dir,
    label_file=None,
    model_type="resnet",
    num_classes=2,
    batch_size=32,
    device="cuda"
):
    # ---------------------
    # Load dataset
    # ---------------------
    dataset = SkinLesionDataset(image_dir=data_dir, label_file=label_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---------------------
    # Load model
    # ---------------------
    model = get_model(model_type=model_type, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    # ---------------------
    # Inference loop
    # ---------------------
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # ---------------------
    # Compute metrics
    # ---------------------
    metrics = compute_metrics(all_preds, all_labels, average="binary" if num_classes == 2 else "macro")
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"{k}:")
            for row in v:
                print(" ", row)
        else:
            print(f"{k}: {v}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained skin lesion classifier")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with evaluation images")
    parser.add_argument("--labels", type=str, default=None, help="Path to label JSON file (optional)")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        label_file=args.labels,
        model_type=args.model_type,
        num_classes=args.num_classes,
        batch_size=args.batch_size
    )
