from __future__ import division
import sys, json, math
import os
import numpy as np
import operator


def load_word2vec(filename):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.

    w2vec = {}
    with open(filename, "r") as f_in:
        for line in f_in:
            line_split = line.replace("\n", "").split()
            w = line_split[0]
            vec = np.array([float(x) for x in line_split[1:]])
            w2vec[w] = vec
    return w2vec


def load_contexts(filename):
    # Returns a dict containing a {word: contextcount} mapping.
    # It loads everything into memory.

    data = {}
    for word, ccdict in stream_contexts(filename):
        data[word] = ccdict
    print "file %s has contexts for %s words" % (filename, len(data))
    return data


def stream_contexts(filename):
    # Streams through (word, countextcount) pairs.
    # Does NOT load everything at once.
    # This is a Python generator, not a normal function.
    for line in open(filename):
        word, n, ccdict = line.split("\t")
        n = int(n)
        ccdict = json.loads(ccdict)
        yield word, ccdict


def cossim_sparse(v1, v2):
    # Take two context-count dictionaries as input
    # and return the cosine similarity between the two vectors.
    # Should return a number beween 0 and 1

    cossim_numerator = 0
    norm_1 = 0
    norm_2 = 0
    for context_key in v1:
        if context_key in v2:
            cossim_numerator += v1[context_key] * v2[context_key]
        norm_1 += v1[context_key] ** 2
    for context_key in v2:
        norm_2 += v2[context_key] ** 2

    return cossim_numerator / (math.sqrt(norm_1) * math.sqrt(norm_2))


def cossim_dense(v1, v2):
    # v1 and v2 are numpy arrays
    # Compute the cosine simlarity between them.
    # Should return a number between -1 and 1

    cossim_value = 0
    for index, value in enumerate(v1):
        cossim_value += value * v2[index]

    return cossim_value / (np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)))


def show_nearest(word_2_vec, w_vec, exclude_w, sim_metric):
    # word_2_vec: a dictionary of word-context vectors. Sparse (dictionary) or Dense (numpy array).
    # w_vec: the context vector of a particular query word `w`. Sparse (dictionary) or Dense (numpy array).
    # exclude_w: the words you want to exclude in the responses. It is a set in python.
    # sim_metric: the similarity metric you want to use. It is a python function
    # which takes two word vectors as arguments.

    cossim_values = {}
    nearest_words = []

    # print word_2_vec
    for word in word_2_vec:
        if word not in exclude_w:
            cossim_values[word] = sim_metric(word_2_vec[word], w_vec)

    for word, cossim_value in sorted(cossim_values.items(), key=operator.itemgetter(1))[::-1][:20]:
        nearest_words.append(word)

    return nearest_words

