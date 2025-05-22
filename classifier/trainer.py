import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from classifier.model import get_model
from dataset.loader import get_dataloader
from evaluation.metrics import compute_metrics


def train_classifier(
        data_dir,
        label_file=None,
        model_type="resnet",
        num_classes=2,
        lr=1e-4,
        batch_size=32,
        num_epochs=20,
        output_dir="./checkpoints",
        device="cuda"
):
    # -----------------------------
    # Setup
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    model = get_model(model_type=model_type, num_classes=num_classes).to(device)
    train_loader = get_dataloader(data_dir, label_file, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        # -----------------------------
        # Metrics & Logging
        # -----------------------------
        avg_loss = running_loss / len(train_loader.dataset)
        metrics = compute_metrics(all_preds, all_labels)

        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", metrics["accuracy"], epoch)
        writer.add_scalar("F1/train", metrics["f1"], epoch)

        # Save best model
        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model.pt"))

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train skin lesion classifier")

    parser.add_argument("--data", type=str, required=True, help="Path to training image folder")
    parser.add_argument("--labels", type=str, default=None, help="Optional path to label JSON file")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=str, default="./checkpoints")

    args = parser.parse_args()

    train_classifier(
        data_dir=args.data,
        label_file=args.labels,
        model_type=args.model_type,
        num_classes=args.num_classes,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=args.out
    )
