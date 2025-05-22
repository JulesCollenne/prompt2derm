import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os

# Default model name (can be overridden via config)
DEFAULT_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

def load_diffusion_model(model_name=DEFAULT_MODEL_NAME, device="cuda"):
    """
    Load the Stable Diffusion pipeline with default or fine-tuned weights.

    Args:
        model_name (str): Hugging Face model ID or local path.
        device (str): Device to load the model on ("cuda" or "cpu").

    Returns:
        pipeline (StableDiffusionPipeline): Text-to-image pipeline.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline = pipeline.to(device)
    pipeline.enable_attention_slicing()  # for memory efficiency
    pipeline.safety_checker = None       # disable NSFW filter for medical domain

    return pipeline

def generate_image_from_prompt(prompt, model=None, device="cuda", guidance_scale=7.5, num_inference_steps=50, seed=None):
    """
    Generate a single image from a text prompt.

    Args:
        prompt (str): Natural language prompt to generate the image.
        model (StableDiffusionPipeline): Preloaded model, or None to load default.
        device (str): Device to run inference on.
        guidance_scale (float): Higher values make image more aligned with prompt.
        num_inference_steps (int): Diffusion steps for sampling.
        seed (int or None): Random seed for reproducibility.

    Returns:
        PIL.Image: Generated image.
    """
    if model is None:
        model = load_diffusion_model(device=device)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    with torch.no_grad():
        image = model(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

    return image

def batch_generate(prompts, output_dir, model=None, device="cuda", overwrite=False):
    """
    Generate a batch of images from a list of prompts and save them to disk.

    Args:
        prompts (dict): Dictionary {id: prompt}.
        output_dir (str): Path to output directory.
        model: Preloaded diffusion model (optional).
        device: "cuda" or "cpu"
        overwrite: Whether to overwrite existing files.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model or load_diffusion_model(device=device)

    for img_id, prompt in prompts.items():
        out_path = os.path.join(output_dir, f"{img_id}.png")
        if os.path.exists(out_path) and not overwrite:
            continue
        image = generate_image_from_prompt(prompt, model=model, device=device)
        image.save(out_path)
