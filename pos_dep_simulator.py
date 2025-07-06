import random

POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', 'NUM', 'INTJ']
DEP_TAGS = ['nsubj', 'obj', 'iobj', 'ccomp', 'advmod', 'obl', 'xcomp', 'amod', 'nmod', 'aux']

def simulate_pos_sequence(sentence, min_len=4, max_len=12):



    seq_len = random.randint(min_len, max_len)
    pos_sequence = random.choices(POS_TAGS, k=seq_len)
    return ' '.join(pos_sequence)


def simulate_dep_sequence(sentence, min_len=4, max_len=12):

    seq_len = random.randint(min_len, max_len)
    dep_sequence = random.choices(DEP_TAGS, k=seq_len)
    return ' '.join(dep_sequence)

if __name__ == "__main__":
    example_sentence = "La vie est belle."
    print("POS Sequence:", simulate_pos_sequence(example_sentence))
    print("Dependency Sequence:", simulate_dep_sequence(example_sentence))
