import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

from utils.io import list_image_files

"""
Usage example:

python diffusion/trainer.py \
    --image_dir ./data/real_images \
    --prompt_file ./data/prompts.json \
    --output_dir ./checkpoints/diffusion \
    --batch_size 4 --num_epochs 10
"""

# -------------------------------
# Dataset for Prompt-Image Pairs
# -------------------------------
class PromptImageDataset(Dataset):
    def __init__(self, image_dir, prompt_file, tokenizer, resolution=512):
        self.image_paths = list_image_files(image_dir)
        with open(prompt_file, 'r') as f:
            self.prompts = json.load(f)

        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image_id = os.path.splitext(os.path.basename(path))[0]
        prompt = self.prompts.get(image_id, "Skin lesion")

        image = Image.open(path).convert("RGB")
        pixel_values = self.transform(image)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

# -------------------------------
# Training Loop
# -------------------------------
def train(
    image_dir,
    prompt_file,
    output_dir="./checkpoints/diffusion",
    model_name="CompVis/stable-diffusion-v1-4",
    batch_size=4,
    num_epochs=10,
    lr=1e-5,
    device="cuda"
):
    os.makedirs(output_dir, exist_ok=True)

    # Load base model components
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # Dataset
    dataset = PromptImageDataset(image_dir, prompt_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    # Training
    unet.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # Save checkpoint
        unet.save_pretrained(os.path.join(output_dir, f"unet_epoch{epoch+1}"))

    print("Training complete. Final model saved.")
