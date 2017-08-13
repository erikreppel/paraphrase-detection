# %%
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        similarities = [synset.path_similarity(ss) for ss in synsets2]
        similarities = list(filter(lambda x: x is not None, similarities))
        if len(similarities) > 0:
            best_score = max(similarities)
        else:
            best_score = None
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    score /= count
    return score


def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2 


def process_sentences(train):
    scores = []
    for s1, s2 in zip(train['str1'], train['str2']):
        result = symmetric_sentence_similarity(s1, s2)
        scores.append(result)
    return np.array(scores)

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

preds = process_sentences(train)
print(accuracy_score(train['paraphrase'], preds))


# np.sum(train['paraphrase'])
# %%
X = np.array([.3,.6,.7])
X.round()