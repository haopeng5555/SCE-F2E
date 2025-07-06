# SCE–F2E–Dataset

This repository folder contains the full bilingual dataset and its preview version for the SCE–F2E–Lite project: a semantically enriched French-to-English dataset designed for context-aware neural machine translation research.

## Folder Structure

SCE_F2E_Dataset
   ├── SCE_F2E_Shortened.csv  
   ├── SCE_F2E_Dataset.json  

## File Descriptions

### 1. SCE_F2E_Shortened.csv
- This is a shortened version of the full dataset containing approximately 1,700 entries.
- It is provided to offer better readability and quick access since large JSON files are not easily viewable in standard text editors or spreadsheets.
- This file includes the following columns:
  - French Sentence
  - English Translation
  - POS Sequence
  - Dependency Sequence
  - Valence
  - Relative Position

This CSV file is ideal for browsing, sampling, demonstrations, and debugging.

### 2. SCE_F2E_Dataset.json
- This is the complete dataset containing 20,000 unique French–English sentence pairs with full linguistic annotations and metadata.
- The JSON structure is designed for direct use in model training pipelines or downstream tasks.
- Each record in the JSON includes:
  - `FrenchSentence`: The original French sentence (string)
  - `EnglishTranslation`: The English translation (string)
  - `POSSequence`: The Part-of-Speech tags for the French sentence (list of strings)
  - `DependencySequence`: The syntactic dependency relations (list of strings)
  - `Valence`: The emotional valence score of the sentence (float, typically between -1 and +1)
  - `RelativePosition`: The relative positional index of the token within the sentence (float between 0 and 1)

This file is intended for programmatic use and model consumption.

## Usage Guidelines

- For quick analysis, visualization, or manual inspection, use the `.csv` file.
- For full-scale model training or research experiments, use the `.json` file.

## Notes

- All data is unique, non-repetitive, and enriched with simulated yet linguistically plausible annotations.
- The dataset is released for research and academic purposes only.

## Citation
If you use this dataset in your work, please cite the original SCE–F2E–Lite paper, which will be added post the publication. 

---

For code to generate or work with this dataset, please refer to the `dataset_generation` folder in this repository.

