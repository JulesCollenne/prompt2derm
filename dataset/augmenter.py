import os
import json
from tqdm import tqdm
from PIL import Image

from diffusion.stable_diffusion_wrapper import load_diffusion_model, generate_image_from_prompt
from utils.io import save_image

def generate_synthetic_dataset(prompt_file, output_dir, diffusion_model=None, overwrite=False):
    """
    Generate synthetic images from descriptions using a diffusion model.
    Args:
        prompt_file (str): Path to a JSON file with {id: prompt} entries.
        output_dir (str): Directory to save generated images.
        diffusion_model: An already loaded model or None to load internally.
        overwrite (bool): Overwrite existing files.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(prompt_file, 'r') as f:
        prompts = json.load(f)

    if diffusion_model is None:
        diffusion_model = load_diffusion_model()

    for key, prompt in tqdm(prompts.items(), desc="Generating synthetic images"):
        out_path = os.path.join(output_dir, f"{key}.png")
        if os.path.exists(out_path) and not overwrite:
            continue
        image = generate_image_from_prompt(prompt, model=diffusion_model)
        save_image(image, out_path)
