# BELLE

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2510.24372-b31b1b.svg)](https://arxiv.org/abs/2510.24372)
[![License](https://img.shields.io/badge/License-See%20LICENSE-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)

This is the **official repository** for the paper: **[Bayesian Speech Synthesizers Can Learn from Multiple Teachers](https://arxiv.org/abs/2510.24372)**.  
This codebase provides training, data preparation, and evaluation pipelines for **BELLE**, and also **reproduces MELLE**.

---

## ✨ Highlights
- End‑to‑end training and evaluation pipelines aligned with the paper.
- Modular data pipeline with optional TTS‑based augmentation.
- Multi‑model training scripts for BELLE / MELLE / BELLE‑stream.
- Zero‑shot TTS evaluation suite under [evaluate-zero-shot-tts](evaluate-zero-shot-tts).

---

## 🧠 Model Overview
BELLE reframes TTS as **Bayesian inference** rather than deterministic regression. It models acoustic targets with a **Normal‑Inverse‑Gamma** distribution to capture data‑dependent aleatoric uncertainty **without increasing parameters or inference latency**. To learn reliable variance from single‑reference datasets, BELLE introduces a **one‑to‑many training strategy** that leverages synthetic samples as a statistical support set. The framework naturally supports **high‑quality streaming generation**.

**Architecture at a glance:**

![BELLE architecture](Figures/belle.png)

**Key contributions (summary):**
- Bayesian evidential learning for continuous AR TTS with uncertainty modeling.
- One‑to‑many training with multiple teachers to improve robustness.
- Strong results with smaller data scale and streaming capability.

---

## 📦 Repository Layout

```
BELLE/
├─ belle/                       # Core library (models, modules, utils)
├─ egs/librispeech/             # Data prep + training entrypoints
├─ evaluate-zero-shot-tts/      # Zero‑shot evaluation suite
├─ tts-launch/                  # TTS augmentation launch scripts
├─ scripts/                     # High‑level run scripts (train/eval/env)
├─ pretrained/                  # Pretrained checkpoints (if any)
└─ Figures/                     # Paper figures (optional)
```

---

## 🧰 Environment Setup
The environment setup script is provided at:

- [scripts/setup.sh](scripts/setup.sh)

It creates a `conda` environment, installs dependencies, pulls `k2`, installs `icefall`, and sets up evaluation dependencies.

> ⚠️ **Note:** The `k2` wheel must match your CUDA and PyTorch versions.  
> See the comment inside [scripts/setup.sh](scripts/setup.sh) for guidance.

---

## 🧪 Data Pipeline (LibriSpeech)
First download LibriSpeech from the official page: **https://www.openslr.org/12** and extract it to:

- [egs/librispeech/download/LibriSpeech](egs/librispeech/download/LibriSpeech) (see Step 0 in the script below)

The end‑to‑end data pipeline is implemented in:

- [egs/librispeech/utils/data_pipeline.sh](egs/librispeech/utils/data_pipeline.sh)

**Overview of steps:**
1. Prepare LibriSpeech manifests and tokenize text (G2P phonemes).
2. Apply VAD and regenerate manifests.
3. Filter by duration.
4. *(Optional)* TTS‑based augmentation using multiple models.
5. VAD on synthesized audio and filter failed items.
6. Prepare training cuts for BELLE / MELLE.
7. Prepare prompt‑augmented cuts for BELLE‑stream.

---

## 🚀 Training
Training is driven by the scripts below (each script loads the matching settings file):

- **BELLE:** [scripts/run_belle.sh](scripts/run_belle.sh) → [scripts/settings_belle.sh](scripts/settings_belle.sh)
- **MELLE:** [scripts/run_melle.sh](scripts/run_melle.sh) → [scripts/settings_melle.sh](scripts/settings_melle.sh)
- **BELLE‑stream:** [scripts/run_stream.sh](scripts/run_stream.sh) → [scripts/settings_stream.sh](scripts/settings_stream.sh)

All training scripts invoke [egs/librispeech/bin/trainer.py](egs/librispeech/bin/trainer.py) with distributed `torchrun`.

**Typical workflow:**
1. Run the data pipeline.
2. Start training with the corresponding run script listed above.
3. For BELLE‑stream, initialize from the last BELLE checkpoint and set `start_epoch=2`, `train_stage=2` (already reflected in [scripts/settings_stream.sh](scripts/settings_stream.sh)).

---

## 📊 Evaluation (Zero‑Shot TTS)
Evaluation is managed under:

- [evaluate-zero-shot-tts/scripts/pipeline_new.sh](evaluate-zero-shot-tts/scripts/pipeline_new.sh)

This pipeline includes **inference + evaluation**, and supports metrics such as:
- **WER** (Hubert / Conformer ASR backends)
- **MOS** (UTMOS)
- **Similarity** (WavLM‑Uni: `sim_o`, `sim_r`)

---

## 📝 Citation
If you use this repository, please cite the paper:

```
@article{zhang2025bayesian,
  title={Bayesian Speech Synthesizers Can Learn from Multiple Teachers},
  author={Zhang, Ziyang and Gao, Yifan and Xu, Xuenan and Wu, Wen and Zhang, Chao and others},
  journal={arXiv preprint arXiv:2510.24372},
  year={2025}
}
```

---

## 📄 License
See [LICENSE](LICENSE).

---

## 🙏 Acknowledgements
We heavily referenced the implementation from **[lifeiteng/vall-e](https://github.com/lifeiteng/vall-e)**.  
This project also builds upon **[icefall](https://github.com/k2-fsa/icefall)** and related open‑source speech toolkits.
