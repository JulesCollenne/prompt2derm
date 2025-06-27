import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


"""
Example usage:

from evaluation.visualizer import visualize_grid
from utils.io import load_json

descriptions = load_json("./data/descriptions.json")
visualize_grid(
    prompt_dict=descriptions,
    real_dir="./data/real_images",
    synth_dir="./data/synthetic_images",
    output_dir="./visualizations",
    max_images=12
)
"""

def draw_triplet(prompt, real_image_path, synthetic_image_path, output_path=None, font_size=16):
    """
    Creates a side-by-side image of [Prompt | Real | Synthetic] for visual inspection.

    Args:
        prompt (str): The text prompt describing the lesion.
        real_image_path (str): Path to the real image file.
        synthetic_image_path (str): Path to the synthetic image file.
        output_path (str): Optional path to save the visualized triplet.
        font_size (int): Font size for prompt text overlay.

    Returns:
        Image: PIL image of the triplet layout.
    """
    real = Image.open(real_image_path).convert("RGB").resize((256, 256))
    synth = Image.open(synthetic_image_path).convert("RGB").resize((256, 256))

    # Optional: load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Add prompt as a blank canvas with text
    prompt_img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(prompt_img)
    draw.multiline_text((10, 10), prompt, fill=(0, 0, 0), font=font, spacing=4)

    # Combine images horizontally
    combined = Image.new("RGB", (256 * 3, 256))
    combined.paste(prompt_img, (0, 0))
    combined.paste(real, (256, 0))
    combined.paste(synth, (512, 0))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.save(output_path)

    return combined


def visualize_grid(prompt_dict, real_dir, synth_dir, output_dir="./visualizations", max_images=10):
    """
    Generate and save N prompt-real-synthetic visualizations.

    Args:
        prompt_dict (dict): {image_id: prompt}
        real_dir (str): Directory containing real images.
        synth_dir (str): Directory containing synthetic images.
        output_dir (str): Directory to save visualizations.
        max_images (int): Number of triplets to generate.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for image_id, prompt in prompt_dict.items():
        real_path = os.path.join(real_dir, f"{image_id}.png")
        synth_path = os.path.join(synth_dir, f"{image_id}.png")

        if not os.path.exists(real_path) or not os.path.exists(synth_path):
            continue

        out_path = os.path.join(output_dir, f"{image_id}_triplet.png")
        draw_triplet(prompt, real_path, synth_path, output_path=out_path)

        count += 1
        if count >= max_images:
            break


def show_image_grid(images, titles=None, ncols=3, figsize=(12, 4)):
    """
    Display a list of images in a matplotlib grid.

    Args:
        images (List[PIL.Image]): List of images to display.
        titles (List[str]): Optional list of titles.
        ncols (int): Number of columns in the grid.
        figsize (tuple): Size of the matplotlib figure.
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].axis("off")
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
