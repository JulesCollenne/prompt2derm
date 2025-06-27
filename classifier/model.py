import torch.nn as nn
import torchvision.models as models

from transformers import ViTForImageClassification


# -------------------------------
# CNN-based Classifier (ResNet50)
# -------------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# --------------------------------------
# ViT-based Classifier (HuggingFace ViT)
# --------------------------------------
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained_model="google/vit-base-patch16-224"):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes
        )

    def forward(self, x):
        return self.model(pixel_values=x).logits

# -------------------------------
# Factory Method
# -------------------------------
def get_model(model_type="resnet", num_classes=2, **kwargs):
    """
    Returns a model instance based on type.
    Args:
        model_type (str): "resnet" or "vit"
        num_classes (int): number of output classes
        **kwargs: additional parameters passed to models
    """
    if model_type == "resnet":
        return ResNetClassifier(num_classes=num_classes, **kwargs)
    elif model_type == "vit":
        return ViTClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
