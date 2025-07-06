import pandas as pd
import random
import os
from transformers import pipeline

# Define input paths (replace with your actual dataset paths)
TED_PATH = 'TED2020.csv'
TATOEBA_PATH = 'Tatoeba.csv'
SUBTITLE_PATH = 'OpenSubtitles.csv'

# Define output path
OUTPUT_PATH = 'core_dataset_output.csv'

# Load Hugging Face emotion classification pipeline (zero-shot model)
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Load datasets
sources = {
    'TED': pd.read_csv(TED_PATH),
    'Tatoeba': pd.read_csv(TATOEBA_PATH),
    'Subtitles': pd.read_csv(SUBTITLE_PATH)
}

# Prepare empty list for storing results
filtered_rows = []

# Function to detect emotion using transformer model
def detect_emotion(text):
    try:
        result = classifier(text[:512])  # Truncate to 512 tokens max
        label = result[0]["label"].lower()
        return label
    except Exception:
        return 'neutral'

# Process each source
for source_name, df in sources.items():
    # Ensure consistent column names: 'source' (French), 'target' (English)
    if 'source' not in df.columns or 'target' not in df.columns:
        continue  # Skip incompatible datasets

    for idx, row in df.iterrows():
        src = str(row['source'])
        tgt = str(row['target'])

        # Skip very short or empty sentences
        if len(src.split()) < 3 or len(tgt.split()) < 3:
            continue

        emotion = detect_emotion(src)

        filtered_rows.append({
            'source': src,
            'target': tgt,
            'emotion': emotion,
            'domain': source_name
        })

# Convert to DataFrame
final_df = pd.DataFrame(filtered_rows)

# Shuffle for randomness
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"Core dataset with {len(final_df)} rows saved to: {OUTPUT_PATH}")
