import numpy as np
import nltk

# This code imports the numpy library for numerical computations and the nltk library for natural language processing.

nltk.download('punkt')
# This line downloads the punkt tokenizer, a tool used for tokenizing text.

from nltk.stem.porter import PorterStemmer
# This line imports the PorterStemmer, a tool used for stemming words.

stemmer = PorterStemmer()
# This initializes the stemmer, which is used to find the root form of words.

def tokenize(sentence):
    """
    This function splits a sentence into individual words or tokens.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    This function finds the root form of a word.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    This function creates a bag of words representation for a given sentence.
    It returns an array where each element corresponds to whether a known word is present in the sentence or not.
    """
    # Stem each word in the tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
            
    return bag
