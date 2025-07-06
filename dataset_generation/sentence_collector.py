import requests
import pandas as pd
import time
import uuid

TATOEBA_API_URL = "https://tatoeba.org/en/api_v0/search"
OUTPUT_FILE = "collected_sentences.csv"

# Configuration: Number of sentences to fetch
TARGET_SENTENCE_COUNT = 2000
BATCH_SIZE = 100  # Number of sentences to fetch per API call

def fetch_tatoeba_sentences(query, from_lang="fra", to_lang="eng", limit=100):
    params = {
        "query": query,
        "from": from_lang,
        "to": to_lang,
        "orphans": "no",
        "has_audio": "no",
        "sort": "words",
        "trans_filter": "limit",
        "trans_has_audio": "no",
        "trans_orphan": "no",
        "limit": limit
    }

    response = requests.get(TATOEBA_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        results = []
        for item in data.get("results", []):
            french_sentence = item.get("sentence", {}).get("text", "").strip()
            english_translations = [
                t.get("text", "").strip() for t in item.get("translations", [])
                if t.get("lang") == to_lang
            ]
            for translation in english_translations:
                if french_sentence and translation:
                    results.append((french_sentence, translation))
        return results
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return []

def generate_sentence_pairs(keywords, target_count):
    all_pairs = set()

    for keyword in keywords:
        print(f"Fetching sentences for keyword: {keyword}")
        fetched_pairs = fetch_tatoeba_sentences(keyword, limit=BATCH_SIZE)

        for pair in fetched_pairs:
            all_pairs.add(pair)

        time.sleep(1)  # To avoid hitting API too fast

        if len(all_pairs) >= target_count:
            break

    return list(all_pairs)[:target_count]

def save_sentences_to_csv(pairs, output_file):
    df = pd.DataFrame(pairs, columns=["French Sentence", "English Translation"])
    df['Sentence ID'] = [str(uuid.uuid4()) for _ in range(len(df))]
    df = df[["Sentence ID", "French Sentence", "English Translation"]]
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved {len(df)} sentence pairs to {output_file}")

if __name__ == "__main__":
    # Broad keyword pool for diverse context extraction
    keywords = [
        "amour", "tristesse", "succès", "peur", "bonheur", "colère", "espoir",
        "travail", "échec", "apprentissage", "famille", "voyage", "avenir",
        "école", "amitié", "confiance", "solitude", "liberté", "nature", "musique"
    ]

    sentence_pairs = generate_sentence_pairs(keywords, TARGET_SENTENCE_COUNT)
    save_sentences_to_csv(sentence_pairs, OUTPUT_FILE)
