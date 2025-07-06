# SCE–F2E–Model

This repository folder contains the full model implementation for the **SCE–F2E–Lite** project: a hybrid multi-branch neural architecture designed for context-aware French-to-English machine translation.

The model combines semantic, syntactic, and structural features through a novel tri-branch encoder and fusion strategy to achieve state-of-the-art translation performance, as proposed in the accompanying research paper.

## Folder Structure

```
SCE_F2E_Model/
├── model.py
├── preprocess_utils.py
├── train_and_finetune.py
└── evaluate_metrics.py
```

## File Descriptions

### 1. model.py

* Defines the **TranslationModel**, which integrates three specialized encoders:

  * **cNODE Encoder**: Captures deep semantic information using Neural ODEs.
  * **Reformer-MoE Encoder**: Handles long-range dependencies efficiently using sparse attention with Mixture-of-Experts.
  * **Capsule Graph Encoder**: Preserves syntactic and grammatical structure using capsule networks.
* Includes a **Fusion Layer** to adaptively combine encoder outputs and a **CAPM-CTC Decoder** for translation generation.

### 2. preprocess\_utils.py

* Provides utilities for loading and preprocessing the **SCE–F2E–Lite** dataset.
* Supports:

  * POS tagging
  * Dependency parsing
  * Valence scoring
  * Relative position extraction
* Includes a PyTorch-compatible dataset class and functions to create training, validation, and test DataLoaders.

### 3. train\_and\_finetune.py

* Implements the complete training loop for the model.
* Supports optional **Minimum Risk Training (MRT)** for sentence-level translation optimization.
* Prints standard classification metrics (accuracy, precision, recall, F1-score) after training.

### 4. evaluate\_metrics.py

* Computes and reports the following translation quality metrics:

  * **BLEU**
  * **METEOR**
  * **BERTScore**
  * **Structural Fidelity Metric (SFM)**
* Can be used to evaluate model predictions against ground-truth translations.

## Usage Guidelines

* Prepare the annotated **SCE–F2E–Lite** dataset and place it in:

```
data/SCE_F2E_Dataset/SCE_F2E_Dataset.json
```

* For data preprocessing and DataLoader creation:

```python
from preprocess_utils import prepare_datasets
train_loader, val_loader, test_loader = prepare_datasets('data/SCE_F2E_Dataset/SCE_F2E_Dataset.json')
```

* To train the model:

```
python train_and_finetune.py
```

* To evaluate model outputs:

```
python evaluate_metrics.py
```

## Notes

* The codebase is implemented in **PyTorch** and supports both CPU and GPU execution.
* All models, datasets, and evaluation scripts are modular and can be extended for future research including multilingual or multimodal translation.

## Citation

If you use this model in your research, please cite the original **SCE–F2E–Lite** paper (citation details will be provided upon publication).

---
