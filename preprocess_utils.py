import json
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple


# Load JSON dataset containing the pre-annotated translation pairs and features
def load_dataset(json_path: str) -> List[dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    return data


# Custom PyTorch Dataset class to handle the translation data
class TranslationDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer_name: str = 'bert-base-multilingual-cased', max_length: int = 64):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text = item['French Sentence']
        target_text = item['English Translation']

        # Tokenize source and target
        source = self.tokenizer(source_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        target = self.tokenizer(target_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': source['input_ids'].squeeze(0),
            'attention_mask': source['attention_mask'].squeeze(0),
            'labels': target['input_ids'].squeeze(0),
            'pos_sequence': item['POS Sequence'],
            'dep_sequence': item['Dependency Sequence'],
            'valence': item['Valence'],
            'rel_position': item['Relative Position']
        }


# Split the dataset into training, validation, and test sets and create DataLoaders
def prepare_datasets(json_path: str, batch_size: int = 16, test_size: float = 0.15, val_size: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data = load_dataset(json_path)

    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=val_size / (1 - test_size), random_state=42)

    train_dataset = TranslationDataset(train_data)
    val_dataset = TranslationDataset(val_data)
    test_dataset = TranslationDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Compute BLEU Score for a given pair of sentences
def compute_bleu(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    return round(bleu_score * 100, 2)


# Compute BERTScore using pretrained multilingual BERT model
def compute_bertscore(candidate: str, reference: str, model_name: str = 'bert-base-multilingual-cased') -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs_candidate = tokenizer(candidate, return_tensors='pt', truncation=True, padding=True)
    inputs_reference = tokenizer(reference, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        candidate_embeddings = model(**inputs_candidate).last_hidden_state.mean(dim=1)
        reference_embeddings = model(**inputs_reference).last_hidden_state.mean(dim=1)

    score = torch.nn.functional.cosine_similarity(candidate_embeddings, reference_embeddings).item()
    return round(score * 100, 2)


# Compute Structural Fidelity Metric (SFM) as proposed in the paper
# For simplicity, we calculate overlap between POS sequences and Dependency sequences
def compute_sfm(pos_seq_true: str, pos_seq_pred: str, dep_seq_true: str, dep_seq_pred: str) -> float:
    pos_true_set = set(pos_seq_true.split())
    pos_pred_set = set(pos_seq_pred.split())
    dep_true_set = set(dep_seq_true.split())
    dep_pred_set = set(dep_seq_pred.split())

    pos_overlap = len(pos_true_set.intersection(pos_pred_set)) / max(len(pos_true_set.union(pos_pred_set)), 1)
    dep_overlap = len(dep_true_set.intersection(dep_pred_set)) / max(len(dep_true_set.union(dep_pred_set)), 1)

    sfm_score = (pos_overlap + dep_overlap) / 2
    return round(sfm_score * 100, 2)


# Compute scaled valence to ensure consistent emotion range normalization
def scale_valence(valence: float) -> float:
    scaled = max(min(valence, 1.0), -1.0)
    return round(scaled, 3)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_datasets("SCE_F2E_Dataset.json", batch_size=8)

    for batch in test_loader:
        inputs = batch['input_ids']
        labels = batch['labels']
        valence = batch['valence']

        print(f"Sample Input IDs: {inputs.shape}")
        print(f"Sample Valence: {valence}")
        break
