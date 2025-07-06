# SCE-F2E
SCE-F2E Project

---

# SCE–F2E–Lite: Context-Aware French–English Neural Machine Translation

This repository contains the complete codebase, dataset, and documentation for the **SCE–F2E–Lite** project: a novel hybrid neural machine translation framework that integrates semantic, syntactic, and discourse-level information for enhanced French-to-English translation. The project also introduces **SCE–F2E–Lite**, a custom bilingual dataset enriched with linguistic annotations.

The proposed system, described in the accompanying research paper (submitted to *IEEE Access*), achieves superior translation accuracy, fluency, and structural fidelity through a multi-branch encoder architecture and sentence-level optimization.

---

## Repository Structure

```
SCE_F2E_Lite/
├── dataset_generation/      # Scripts to generate the SCE–F2E–Lite dataset
├── SCE_F2E_Dataset/         # Pre-generated full dataset and preview version
└── model_implementation/    # Code for model training, evaluation, and metrics
```

---

## 1. Dataset Generation (`dataset_generation/`)

This folder contains scripts to create the **SCE–F2E–Lite** dataset from scratch, including:

* Sentence collection
* Valence annotation
* POS and dependency parsing
* Optional emotion refinement
* Final dataset assembly in CSV and JSON formats

Refer to the detailed instructions in `dataset_generation/README.md`.

---

## 2. Dataset Files (`SCE_F2E_Dataset/`)

This folder provides:

* `SCE_F2E_Shortened.csv` — A lightweight preview (1,700 entries) for quick exploration
* `SCE_F2E_Dataset.json` — The full dataset (20,000 entries) with:

  * French and English sentences
  * Part-of-Speech sequences
  * Dependency sequences
  * Valence scores
  * Relative token positions

Refer to `SCE_F2E_Dataset/README.md` for file descriptions and usage guidelines.

---

## 3. Model Implementation (`model_implementation/`)

This folder contains the complete implementation of the proposed translation framework, including:

* The hybrid **cNODE–Reformer–Capsule** model (`model.py`)
* Data preprocessing utilities (`preprocess_utils.py`)
* Training and fine-tuning scripts (`train_and_finetune.py`)
* Evaluation scripts for computing BLEU, METEOR, BERTScore, and Structural Fidelity (`evaluate_metrics.py`)

Detailed instructions are provided in `model_implementation/README.md`.

---

## Key Features of the Project

* **Tri-Branch Hybrid Architecture**:

  * Semantic modeling using Neural ODEs
  * Sparse attention with Reformer and Mixture-of-Experts
  * Syntactic encoding with Capsule Networks

* **Hyper-Gated Fusion Layer** for dynamic context-sensitive weighting

* **CAPM-CTC Decoder** with optional **Minimum Risk Training** for sentence-level optimization

* **Custom Evaluation Metrics** including BLEU, METEOR, BERTScore, and Structural Fidelity Metric (SFM)

* **Open Bilingual Dataset** enriched with linguistic and affective annotations

---

## Getting Started

1. **Generate the Dataset** (optional if using pre-generated files):

```
cd dataset_generation
python sentence_collector.py
python valence_annotator.py
python pos_dep_simulator.py
python data_generator.py
```

2. **Preprocess Data and Train Model**:

```
cd model_implementation
python train_and_finetune.py
```

3. **Evaluate Model**:

```
python evaluate_metrics.py
```

---

## Requirements

* Python 3.8+
* PyTorch
* scikit-learn
* transformers (HuggingFace)
* nltk
* spaCy (`fr_core_news_md` model)

---

## Notes

* The dataset and code are modular and can be extended to other language pairs or multimodal translation tasks.
* All components are released for **research and academic use only**.

---

## Citation

If you use this work, please cite the forthcoming **SCE–F2E–Lite** paper (to be added post-publication).

---

