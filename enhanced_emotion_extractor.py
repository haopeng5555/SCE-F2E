import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import csv

# Load pretrained emotion detection model (context-sensitive)
MODEL_NAME = 'cardiffnlp/twitter-roberta-base-emotion'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Define emotion labels used by the model
EMOTION_LABELS = ['anger', 'joy', 'optimism', 'sadness']

# Load your core dataset (input file path)
input_csv = 'all_data.csv'
df = pd.read_csv(input_csv)

# Function to detect dominant emotion using the transformer model
def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
    top_idx = np.argmax(scores)
    return EMOTION_LABELS[top_idx], scores[top_idx]

# Apply emotion detection to both French and English texts
results = []

for idx, row in df.iterrows():
    try:
        fr_text = str(row['French']).strip()
        en_text = str(row['English']).strip()

        # Skip short sentences
        if len(fr_text.split()) < 4 or len(en_text.split()) < 4:
            continue

        fr_emotion, fr_conf = detect_emotion(fr_text)
        en_emotion, en_conf = detect_emotion(en_text)

        # Only keep pairs where both sides have high-confidence emotions (optional threshold)
        if fr_conf > 0.7 or en_conf > 0.7:
            results.append({
                'French': fr_text,
                'English': en_text,
                'French_Emotion': fr_emotion,
                'French_Confidence': round(fr_conf, 3),
                'English_Emotion': en_emotion,
                'English_Confidence': round(en_conf, 3)
            })

    except Exception as e:
        continue

# Save filtered dataset
output_csv = 'emotion_filtered_dataset.csv'
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Filtered dataset saved: {output_csv} ({len(results)} rows)")
