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
```

# 📦 Setup

## Requirements

- Python 3.10+
- CUDA-enabled GPU (for diffusion and classification)
- PyTorch ≥ 2.0
- Transformers, diffusers, OpenAI API (for LLMs)

## Installation

```bash
git clone https://github.com/yourusername/prompt2derm.git
cd prompt2derm
pip install -r requirements.txt
```

Configure your OpenAI or HuggingFace keys in `.env` or `config.py`.

---

# 🧪 Usage

## 1. Generate Descriptions of Real Images using Image-to-Text Model 

```bash
python descriptions/generate_descriptions.py --input_dir ./data/real_images/ --output ./data/descriptions.json
```

## 2. Fine-Tune Stable Diffusion with obtained Descriptions and Real Images

```bash
python diffusion/trainer.py --descriptions ./data/descriptions.json --images ./data/real_images/
```

## 3. Generate Synthetic Images using Random Prompts

```bash
python diffusion/stable_diffusion_wrapper.py --prompt_file ./data/descriptions.json --output_dir ./data/synthetic_images/
```

## 4. Train Classifier over Augmented Dataset

```bash
python classifier/trainer.py --data ./data/mixed_dataset/
```

## 5. Evaluate Classifier

```bash
python classifier/evaluator.py --model_path ./checkpoints/model.pt
```

---

# 📊 Evaluation Metrics

- **Visual Fidelity**: FID, CLIP similarity  
- **Prompt-Image Alignment**: CLIP score  
- **Classification**: Accuracy, F1-score, ROC-AUC  
- **Fairness**: Sensitivity by skin tone or lesion type  

---

# 📈 Roadmap

- [x] Prompt generation from dermoscopic images  
- [x] Fine-tuning diffusion models  
- [x] Structured prompt templates  
- [ ] Dermatologist-in-the-loop prompt validation  
- [ ] Curriculum generation via classifier feedback  
- [ ] Web interface for interactive prompt-to-image synthesis  

---

# 🤝 Citation

If you use this work, please cite the upcoming paper:

> *Prompt-Driven Diffusion for Dermatological Image Synthesis: A Natural Language Interface to Realistic and Diverse Melanoma Generation*  
> Author Names, MICCAI 2025 (submitted)

---

# 📬 Contact

For questions or collaboration:  
**Rabih Chamas** – [you@example.com](mailto:)

**Jules Collenne** [jules.collenne@gmail.com](mailto:jules.collenne@gmail.com)

*Laboratoire Informatique et Systèmes*
*Aix-Marseille University*