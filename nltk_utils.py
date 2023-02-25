import nltk

from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

# function 1: tokenization of training data


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# function 2: stemming


def stem(word):
    return stemmer.stem(word.lower())


# function 3: creating bag of words
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[i] = 1.0
    return bag
