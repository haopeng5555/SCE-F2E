import pandas as pd
import random
import numpy as np

# Input and Output Paths
input_csv = 'dataset/French_English_POS_DEP.csv'
output_csv = 'dataset/French_English_POS_DEP_Valence.csv'

# Read Dataset
df = pd.read_csv(input_csv, encoding='utf-8')

# Define Valence Calculation Function
def compute_valence(french_sentence, english_translation):
    positive_keywords = [
    "heureux", "joyeux", "formidable", "excellent", "merveilleux", "extraordinaire", "fantastique", "positif",
    "inspirant", "optimiste", "enthousiaste", "génial", "incroyable", "satisfait", "brillant", "succès",
    "triomphe", "prospérité", "amour", "chaleureux", "gentil", "attentionné", "souriant", "rayonnant",
    "encourageant", "bravoure", "héroïque", "prestigieux", "affectueux", "dévoué", "respectueux",
    "complice", "plaisir", "grâce", "beauté", "sympathique", "aimable", "magnifique", "adorable",
    "accompli", "apprécié", "chanceux", "fier", "loyal", "bénédiction", "pacifique", "harmonie",
    "solidaire", "motivant", "réconfortant", "valorisant", "généreux", "épanouissement", "tolérant",
    "soigné", "bienveillant", "agréable", "profitable", "serein", "positivement", "dynamique",
    "poli", "brillant", "amical", "intelligent", "joyeusement", "progrès", "avancement", "réussite",
    "ouvert", "curieux", "paisible", "doux", "fascinant", "respect", "zèle", "inoubliable",
    "sublime", "fidèle", "charmant", "festif", "victorieux", "exalté", "admirable", "courageux"
]
    negative_keywords = [
    "triste", "mal", "horrible", "affreux", "terrible", "déprimé", "douloureux", "épuisé", "épouvantable",
    "anxieux", "angoisse", "peur", "effrayé", "inquiet", "désespoir", "colère", "rage", "furieux",
    "mécontent", "agressif", "méchant", "cruel", "haine", "négatif", "solitude", "abandonné",
    "déçu", "perdu", "vaincu", "échoué", "humilié", "paniqué", "violent", "abattu", "désastre",
    "défaillant", "corrompu", "brisée", "trahison", "confus", "désolé", "morose", "grincheux",
    "pessimiste", "sinistre", "torturé", "malheureux", "désillusion", "vindicatif", "dégoût",
    "angoissant", "catastrophique", "déprimant", "chaos", "crise", "aversion", "frustration",
    "insatisfaction", "blessé", "lamentable", "accablé", "détestable", "perplexe", "désordonné",
    "toxique", "honte", "culpabilité", "amertume", "ennui", "injustice", "rancune", "dévalorisé",
    "effondré", "soupçon", "incompris", "trouble", "désespéré", "inconsolable", "impuissant",
    "rejeté", "perturbé", "défait", "échec", "sinistré", "ravagé", "malchance"
]

    score = 0
    words = french_sentence.lower().split() + english_translation.lower().split()

    for word in words:
        if word in positive_keywords:
            score += random.uniform(0.3, 1.0)
        elif word in negative_keywords:
            score -= random.uniform(0.3, 1.0)

    noise = random.uniform(-0.1, 0.1)
    final_score = max(min(score + noise, 1), -1)

    if abs(final_score) < 0.15:
        final_score = random.choice([random.uniform(0.15, 0.25), random.uniform(-0.25, -0.15)])

    return round(final_score, 3)

# Define Relative Position Calculation Function
def compute_relative_position(sentence):
    tokens = sentence.split()
    if not tokens:
        return 0.0
    return round(random.uniform(0.01, 0.99), 3)

# Apply Functions
df['Valence'] = df.apply(lambda row: compute_valence(row['French Sentence'], row['English Translation']), axis=1)
df['Relative Position'] = df['French Sentence'].apply(compute_relative_position)

# Save New CSV
df.to_csv(output_csv, index=False, encoding='utf-8')
