from nltk import tokenize
import numpy as np


def sentence_to_vec(sentence, vectors, sentence_len):
    '''
    Create word vectors a word vectors for each word in a sentence and add
    empty word vectors so all our sentence matrices have the same length.
    '''
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    words = []
    for token in tokens:
        if token in vectors:
            words.append(vectors[token])

    vec_size = words[0].shape
    while len(words) < sentence_len:
        words.append(np.zeros(vec_size))

    return np.stack(words)


def prep_sentence_couples(sentences1, sentences2, word_vectors):
    '''
    Find the max number of words in a sentence then create sentence matrixes
    for each sentence. Returns have shape (# sentences, max # words, 300)
    '''
    max_word_count = 0
    for s in sentences1:
        tokenizer = tokenize.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(s)
        if len(tokens) > max_word_count:
            max_word_count = len(tokens)

    for s in sentences2:
        tokenizer = tokenize.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(s)
        if len(tokens) > max_word_count:
            max_word_count = len(tokens)

    print('Max word count:', max_word_count)

    vecs1 = [sentence_to_vec(x, word_vectors, max_word_count)
             for x in sentences1]
    vecs2 = [sentence_to_vec(x, word_vectors, max_word_count)
             for x in sentences2]
    return np.stack(vecs1), np.stack(vecs2)
