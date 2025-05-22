import os
import json
import argparse

import torch
from tqdm import tqdm
from PIL import Image

from utils.io import list_image_files, save_json
from prompts.template_utils import make_structured_prompt

# Optional: OpenAI LLM
try:
    import openai
    openai.api_key_path = os.getenv("OPENAI_API_KEY_PATH", ".env")
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def generate_prompt_from_image(image_path, method="openai", model="gpt-4"):
    """
    Generate a natural language prompt from a dermoscopic image.

    Args:
        image_path (str): Path to the input image.
        method (str): 'template', 'openai'
        model (str): OpenAI model name

    Returns:
        str: Prompt describing the lesion.
    """
    if method == "template":
        return make_structured_prompt(image_path)  # Uses heuristics/templates

    elif method == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed.")

        # Optional: convert image to base64 if using vision model
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        # Replace with your custom prompt instructions
        system_msg = "You are a medical assistant. Describe the skin lesion image in clinical terms (color, border, texture, artifacts)."
        user_msg = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data.encode('base64')}"}}  # pseudo code

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content'].strip()

    elif method == "local":
        from transformers import BlipProcessor, BlipForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100)
            caption = processor.decode(out[0], skip_special_tokens=True)

        # Optionally rephrase or add clinical tokens here
        return f"Describe a skin lesion with: {caption}"
    else:
        raise ValueError(f"Unknown method: {method}")


def generate_prompt_dataset(image_dir, output_path, method="template", limit=None):
    """
    Generate prompts for all images in a directory and save them to a JSON file.

    Args:
        image_dir (str): Directory of dermoscopic images.
        output_path (str): Where to save the JSON prompt file.
        method (str): 'template' or 'openai'
        limit (int): Max number of images to process (for testing/debug).
    """
    image_paths = list_image_files(image_dir)
    if limit:
        image_paths = image_paths[:limit]

    prompt_dict = {}

    for path in tqdm(image_paths, desc=f"Generating prompts [{method}]"):
        image_id = os.path.splitext(os.path.basename(path))[0]
        try:
            prompt = generate_prompt_from_image(path, method=method)
            prompt_dict[image_id] = prompt
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue

    save_json(prompt_dict, output_path)
    print(f"Saved {len(prompt_dict)} prompts to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts from dermoscopic images")
    parser.add_argument("--input_dir", required=True, help="Directory of dermoscopic images")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--method", default="template", choices=["template", "openai"], help="Prompt generation method")
    parser.add_argument("--limit", type=int, default=None, help="Max number of images to process")

    args = parser.parse_args()

    generate_prompt_dataset(
        image_dir=args.input_dir,
        output_path=args.output,
        method=args.method,
        limit=args.limit
    )
