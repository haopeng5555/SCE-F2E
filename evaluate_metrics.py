
# evaluate_metrics.py

import json
import numpy as np
import evaluate

from typing import List, Tuple

# Load evaluation metrics
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")


def load_predictions_and_references(file_path: str) -> Tuple[List[str], List[str]]:

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = [item["Predicted"] for item in data]
    references = [item["Reference"] for item in data]

    return predictions, references


def compute_bleu(predictions: List[str], references: List[str]) -> float:

    references_list = [[ref.split()] for ref in references]
    predictions_list = [pred.split() for pred in predictions]

    results = bleu.compute(predictions=predictions_list, references=references_list)
    return results["bleu"] * 100


def compute_bertscore(predictions: List[str], references: List[str]) -> float:

    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    return np.mean(results["f1"]) * 100


def compute_meteor(predictions: List[str], references: List[str]) -> float:

    results = meteor.compute(predictions=predictions, references=references)
    return results["meteor"] * 100


def compute_sfm(bleu_score: float, meteor_score: float, bert_score: float) -> float:

    return (bleu_score + meteor_score + bert_score) / 3


if __name__ == "__main__":

    output_file = "output/generated_translations.json"

    predictions, references = load_predictions_and_references(output_file)

    bleu_score = compute_bleu(predictions, references)
    meteor_score = compute_meteor(predictions, references)
    bert_score = compute_bertscore(predictions, references)
    sfm_score = compute_sfm(bleu_score, meteor_score, bert_score)

    print(f"BLEU Score     : {bleu_score:.2f}")
    print(f"METEOR Score   : {meteor_score:.2f}")
    print(f"BERTScore (F1) : {bert_score:.2f}")
    print(f"SFM Score      : {sfm_score:.2f}")
