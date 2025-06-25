# 🧭 Project Roadmap – Prompt2Derm

---

## 🚧 Implementation Goals

- [ ] **Prompt Generation**
  - [x] Generate prompts using GPT-4 vision API
  - [ ] Implement local prompt generation using BLIP or LLaVA
  - [x] Normalize prompt structure into JSON templates
  - [ ] Train SVM or XGBoost on generated prompts
  - [ ] Add dermatologist review layer (optional filtering or scoring)

- [ ] **Diffusion Pipeline**
  - [ ] Load Stable Diffusion components (UNet, VAE, tokenizer)
  - [ ] Fine-tune on prompt-image pairs
  - [ ] Add checkpointing and resume training support
  - [ ] Add mixed-precision / Accelerate / DDP support for fast training

- [ ] **Synthetic Image Generation**
  - [ ] Generate a large-scale dataset from prompts
  - [ ] Implement prompt sampling strategies (jittering, feature-based sampling)
  - [ ] Filter low-quality generations using CLIP scores or prompt similarity

- [ ] **Classifier Training**
  - [ ] Train baseline classifier on real images
  - [ ] Train classifier on real + synthetic images
  - [ ] Evaluate performance on ambiguous / rare classes

- [ ] **Evaluation Tools**
  - [ ] Real vs synthetic image visualizer
  - [ ] Metric scripts for FID, CLIP similarity, classification accuracy
  - [ ] Add feature-space coverage and diversity metrics
  - [ ] Add fairness metrics across lesion types or skin tones

---

## 🧪 Experimental Goals

- [ ] **Ablation Studies**
  - [ ] Prompt generation method: template vs LLM
  - [ ] Prompt structure: clinical JSON vs free text
  - [ ] Synthetic dataset size: 1k / 5k / 10k
  - [ ] Real-to-synthetic ratio impact on classification
  - [ ] Prompt-image alignment and classification correlation

- [ ] **Benchmarks**
  - [ ] Baseline classifier (real only)
  - [ ] Real + synthetic (random prompts)
  - [ ] Real + synthetic (prompt-conditioned, structured prompts)

- [ ] **Optional Analyses**
  - [ ] Class-wise performance analysis
  - [ ] Expert rating of realism and diagnostic utility

---

## 📝 Writing Goals

- [ ] Abstract
- [ ] Introduction
- [ ] Related Work
- [ ] Method (3.1–3.2: Overview, Prompt Generation)
- [ ] Method (3.3: Diffusion Fine-Tuning)
- [ ] Method (3.4: Image Generation & Classification)
- [ ] Experiments (Results, Tables, Ablations)
- [ ] Discussion
- [ ] Conclusion

- [ ] **Figures**
  - [ ] Pipeline diagram (prompt → image → classifier)
  - [ ] Prompt + image examples
  - [ ] Real vs synthetic comparison grid
  - [ ] Classification performance charts
  - [ ] Prompt diversity or CLIP alignment plots

---

## 📦 Optional Extensions

- [ ] Release synthetic dataset on Hugging Face
- [ ] Build Streamlit app / web demo for interactive prompt-to-image
- [ ] Upload code and preprints to GitHub + arXiv

