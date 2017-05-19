from __future__ import division
import random
import copy


def weighted_draw_from_dict(prob_dict):
    # Utility function -- do not modify
    # Randomly choose a key from a dict, where the values are the relative probability weights.
    # http://stackoverflow.com/a/3679747/86684
    choice_items = prob_dict.items()
    total = sum(w for c, w in choice_items)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choice_items:
        if upto + w > r:
            return c
        upto += w
    assert False, "Shouldn't get here"


# ---------------------- write your answers below here -------------------


def draw_next_word_unigram_model(uni_counts):
    unigram_probabilities = {}
    total_count = sum(uni_counts.values())

    for token in uni_counts:
        unigram_probabilities[token] = uni_counts[token] / total_count

    return weighted_draw_from_dict(unigram_probabilities)


def draw_next_word_bigram_model(uni_counts, bi_counts, prev_word):
    bigram_probabilities = {}

    for next_word in bi_counts[prev_word]:
        bigram_probabilities[prev_word + next_word] = bi_counts[prev_word][next_word] / uni_counts[prev_word]

    return weighted_draw_from_dict(bigram_probabilities)


def sample_sentence(uni_counts, bi_counts):
    tokens = ['**START**']

    while not tokens[-1] == '**END**':
        new_token = draw_next_word_bigram_model(uni_counts, bi_counts, tokens[-1])[len(tokens[-1]):]
        tokens.append(new_token)

    return tokens
