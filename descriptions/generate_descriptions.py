import os
import base64
import openai
import json
from PIL import Image
import io
import re

from tqdm import tqdm

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output

    with open(output_file, "w", encoding="utf-8") as out_f:
        for class_folder in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            for filename in tqdm(os.listdir(class_path), desc=f"Processing {class_folder}"):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_path = os.path.join(class_path, filename)
                try:
                    result = generate_dermatology_description(image_path)
                    entry = {
                        "image": os.path.join(class_folder, filename),
                        "class": class_folder,
                        "description": result["feature_vector"].get("description", ""),
                        "features": result["feature_vector"].get("features", {})
                    }
                    out_f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    print(f"Failed on {filename}: {e}")


def generate_dermatology_description(image_path):
    """
    Generates a multi-step dermatology analysis from an image using OpenAI GPT-4o.
    Focused on lesion description, artifacts (no diagnosis).

    Args:
        image_path (str): Path to the skin lesion image.

    Returns:
        dict: Contains description, artifacts, generation prompt, and prototype label.
    """
    # Load API key from environment variable
    api_key = os.getenv("Rabih_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Rabih_OPENAI_API_KEY environment variable not set.")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Load the image from file
    image = Image.open(image_path)

    # 🔁 Save to bytes buffer
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # 🔐 Encode to base64
    image_base64 = base64.b64encode(image_bytes).decode()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a dermatologist AI assistant. Your task is to quantify visual features of skin lesions from dermoscopic images."
                "Rate each feature on a scale from 0.0 to 1.0, based on typical dermatological presentations:\n"
                "- 0.0 = feature is absent or typical of benign lesions\n"
                "- 0.5 = ambiguous or moderately present\n"
                "- 1.0 = strongly present, highly atypical for benign lesions\n"
                "Additionally, provide a concise but clinically rich natural-language description of the lesion.\n"
                "Do not make a diagnosis. Focus only on visual inspection."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                        "Given this dermatoscopic image, evaluate and provide a JSON object with two keys:\n"
                        "1. 'description' – a paragraph summarizing the lesion's dermoscopic appearance\n"
                        "2. 'features' – an object with the following fields (scored 0.0–1.0):\n"
                        "- atypical_pigment_network\n"
                        "- blue_whitish_veil\n"
                        "- atypical_vascular_pattern\n"
                        "- irregular_dots_globules\n"
                        "- irregular_streaks\n"
                        "- irregular_blotches\n"
                        "- regression_structures\n"
                        "- Asymmetry\n"
                        "- Border Irregularity\n"
                        "- light_brown\n"
                        "- dark_brown\n"
                        "- black\n"
                        "- red\n"
                        "- white\n"
                        "- Color Variation\n"
                        "- lesion_shape_multicomponent\n"
                        "- Evolving\n"
                        "- background_skin_contrast\n"
                        "- background_skin_tone\n"
                        "- hair_amount\n"
                        + "- ruler_presence\n"
                        + "- ink_markings\n"
                          "Consider that most lesions are benign nevi. Only assign high values (close to 1.0) if the feature is markedly atypical compared to average dermoscopic images. Return only the JSON object with 'description' and 'features'."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ]
        }
    ]

    # Call GPT-4o with the full conversation
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # gpt-4.1-mini
        messages=messages,
        max_tokens=1500,
        temperature=0.0,
    )
    # Extract the content from the response
    reply_content = response.choices[0].message.content

    # Remove markdown code blocks if present
    clean_reply = re.sub(r"^```(?:json)?\n|\n```$", "", reply_content.strip(), flags=re.MULTILINE)

    try:
        feature_vector = json.loads(clean_reply)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON response.")
        print("Raw content:", reply_content)
        feature_vector = {}

    usage = response.usage
    return {
        "feature_vector": feature_vector,
        "raw_reply": reply_content,
        "usage": usage
    }


if __name__ == '__main__':
    main()
