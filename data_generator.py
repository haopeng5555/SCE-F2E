import pandas as pd
import random
import numpy as np
from valence_annotator import compute_valence

# Function to generate POS sequence
def generate_pos_sequence(num_tokens):
    pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', 'NUM', 'INTJ']
    return ' '.join(random.choices(pos_tags, k=num_tokens))

# Function to generate Dependency sequence
def generate_dependency_sequence(num_tokens):
    dependency_tags = ['nsubj', 'obj', 'iobj', 'ccomp', 'xcomp', 'obl', 'advmod', 'amod', 'cc', 'conj']
    return ' '.join(random.choices(dependency_tags, k=num_tokens))

# Function to compute relative position
def compute_relative_positions(num_tokens):
    return round(random.uniform(0.0, 1.0), 3)

# Main data generation function
def generate_dataset(input_sentences, output_file, num_rows):
    french_sentences = input_sentences['French']
    english_sentences = input_sentences['English']

    rows = []

    for idx in range(min(num_rows, len(french_sentences))):
        fr_sent = french_sentences[idx]
        en_sent = english_sentences[idx]

        num_tokens = len(fr_sent.split())

        pos_seq = generate_pos_sequence(num_tokens)
        dep_seq = generate_dependency_sequence(num_tokens)
        valence = compute_valence(fr_sent)
        rel_pos = compute_relative_positions(num_tokens)

        rows.append({
            'French Sentence': fr_sent,
            'English Translation': en_sent,
            'POS Sequence': pos_seq,
            'Dependency Sequence': dep_seq,
            'Valence': valence,
            'Relative Position': rel_pos
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    generate_dataset(example_input, 'generated_dataset.xlsx', num_rows=4)
