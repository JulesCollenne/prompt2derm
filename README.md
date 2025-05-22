# 🧬 prompt2derm
**Prompt-Driven Diffusion for Dermatological Image Synthesis**

A modular pipeline for generating realistic and diverse dermoscopic skin lesion images from natural language prompts using diffusion models. This project combines large language models (LLMs) and Stable Diffusion to enhance melanoma detection by augmenting datasets with semantically rich synthetic images.

---

## 🚀 Overview

`prompt2derm` addresses the challenges of class imbalance and limited diversity in skin lesion datasets by:

- Generating structured prompts from dermoscopic images using an LLM.
- Fine-tuning Stable Diffusion on prompt–image pairs.
- Synthesizing new lesion images from natural language descriptions.
- Augmenting classifier training with synthetic data to improve robustness.

---

## 🏗️ Project Structure

```text
prompt2derm/
├── main.py                          # Entry point: full pipeline execution
├── config.py                        # Configuration: paths, models, hyperparameters

├── prompts/
│   ├── prompt_generator.py         # Generates prompts from images using an LLM
│   └── template_utils.py           # Slot-based and structured prompt templates

├── diffusion/
│   ├── stable_diffusion_wrapper.py # Sampling and inference from Stable Diffusion
│   └── trainer.py                  # Fine-tuning on prompt–image pairs

├── dataset/
│   ├── loader.py                   # Loads and preprocesses datasets
│   └── augmenter.py                # Generates synthetic images from prompts

├── classifier/
│   ├── model.py                    # CNN / ViT model definitions
│   ├── trainer.py                  # Train classifier on real + synthetic data
│   └── evaluator.py               # Evaluation: metrics, fairness, confusion matrix

├── evaluation/
│   ├── metrics.py                  # FID, CLIP similarity, realism scoring
│   └── visualizer.py              # Visual comparison of real vs synthetic outputs

├── utils/
│   ├── logging.py                 # Logger, wandb/tensorboard hooks
│   └── io.py                      # I/O utilities: saving images, reading JSON, etc.

└── README.md
