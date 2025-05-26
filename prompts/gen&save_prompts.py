import os
import json
from tqdm import tqdm  
from generate_prompts import generate_dermatology_multi_turn

input_dir = "data/valid_split"
output_file = "lesion_features_1.jsonl"

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
                result = generate_dermatology_multi_turn(image_path)
                entry = {
                    "image": os.path.join(class_folder, filename),
                    "class": class_folder,
                    "description": result["feature_vector"].get("description", ""),
                    "features": result["feature_vector"].get("features", {})
                }
                out_f.write(json.dumps(entry) + "\n")
            except Exception as e:
                print(f"Failed on {filename}: {e}")
