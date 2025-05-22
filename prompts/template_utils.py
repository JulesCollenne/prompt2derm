import os
import random
from PIL import Image

# Optional feature categories (can be expanded)
COLORS = ["light brown", "dark brown", "black", "pink", "red", "white", "blue"]
TEXTURES = ["smooth", "rough", "scaly", "crusty", "heterogeneous"]
SHAPES = ["round", "oval", "irregular", "asymmetrical"]
ARTIFACTS = ["fine hairs", "glare", "ink markings", "ruler markings", "no visible artifacts"]

def make_structured_prompt(image_path=None, seed=None):
    """
    Generates a pseudo-clinical prompt describing a lesion using fixed categories.
    Useful as a fallback or baseline method.

    Args:
        image_path (str): Optional path to image (not used unless custom logic is added).
        seed (int): Optional seed for deterministic output.

    Returns:
        str: Natural language prompt describing the lesion.
    """
    if seed is not None:
        random.seed(seed)

    color = ", ".join(random.sample(COLORS, k=random.randint(1, 3)))
    texture = random.choice(TEXTURES)
    shape = random.choice(SHAPES)
    artifact = random.choice(ARTIFACTS)

    prompt = (
        f"Generate an image of a skin lesion with a {shape} shape and a {texture} texture. "
        f"The lesion should have the following colors: {color}. "
        f"The background is skin-toned with {artifact}. "
        f"There should be no rulers, glare, or other distracting elements."
    )

    return prompt
