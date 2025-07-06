# Dataset Generation - SCE–F2E–Lite

This directory contains the scripts for generating the SCE–F2E–Lite dataset used for training context-aware French-to-English neural machine translation models. The dataset is annotated with linguistic, structural, and affective metadata, and is generated through a step-by-step modular pipeline.

## Directory Structure

```
dataset_generation/
├── sentence_collector.py
├── valence_annotator.py
├── pos_dep_simulator.py
├── enhanced_emotion_extractor.py
└── data_generator.py
```

## Overview of Files

### 1. sentence_collector.py
Fetches real French-English sentence pairs using online resources like the Tatoeba API. It filters the data and stores it in a raw CSV format.
- Input: None (connects to external APIs)
- Output: CSV with columns:
  - Sentence ID
  - French Sentence
  - English Translation

### 2. valence_annotator.py
Assigns valence scores to each French sentence based on emotional keywords and lexicons. The scores range from -1.0 (negative) to +1.0 (positive).
- Input: CSV from sentence_collector.py
- Output: Same CSV with an additional column:
  - Valence

### 3. pos_dep_simulator.py
Generates Part-of-Speech (POS) tags and Dependency Relations for each French sentence using spaCy’s fr_core_news_md model.
- Input: CSV from valence_annotator.py
- Output: CSV with two new columns:
  - POS Sequence
  - Dependency Sequence

### 4. enhanced_emotion_extractor.py
Optionally improves the valence annotations using more advanced emotion detection techniques, such as external emotion lexicons or pretrained models.
- Input: CSV from pos_dep_simulator.py
- Output: CSV with refined Valence values

### 5. data_generator.py
Finalizes the dataset by computing the relative positional index of each token and exporting the dataset into both CSV and JSON formats.
- Input: Fully annotated CSV
- Output: Two files:
  - Final CSV file
  - Final JSONL or JSON file

## Code Execution Flow

1. Run `sentence_collector.py` to collect sentence pairs.
2. Run `valence_annotator.py` to assign valence scores.
3. Run `pos_dep_simulator.py` to generate POS and dependency tags.
4. (Optional) Run `enhanced_emotion_extractor.py` to refine valence annotations.
5. Run `data_generator.py` to compute relative position and export the final dataset.

## Example Output Schema

French Sentence, English Translation, POS Sequence, Dependency Sequence, Valence, Relative Position

## Final Notes

- All scripts are modular 
- Outputs are compatible with downstream translation models.
- All scripts assume UTF-8 encoded input files.
- Make sure to install the required Python libraries (e.g., pandas, requests, spacy, etc.) before running the scripts.
