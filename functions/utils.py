#util functions
# Functions for preprocessing
import re
import numpy as np
import nltk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import keras.backend as K

def clean_text(text):
    ''' Pre process and convert texts to a list of words 
        Args:
             text: input text'''
    text = str(text)
    #text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    text = BeautifulSoup(text, "lxml").text 
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    return text

def load_embedding(file_name):
    """Function to create the embedding variable from the file provided
        Args:
            file_name = full path to the file to embedding file"""
    embeddings = {}
    with open(file_name, "r", encoding = 'utf8') as embed:
        for line in embed:
            name, vector = tuple(line.split(" ", 1))
            embeddings[name] = np.fromstring(vector, sep=" ")
    return embeddings

def create_embedding_weights(embeddings, embedding_dim, word_index, NUM_WORDS):
    """Function to create the embedding weights corresponding to the word indices created
        Args:
            embedding: vriable with the word embeddings from a pretrained model
            embedding_dim: dimensions N 
            word_index: dictionary with the word and corresponding indices
            NUM_WORDS: number of words in the dictionary
    """
    #embedding_dim = embeddings['the'].shape[0]
    word_len = min(NUM_WORDS, len(word_index))
    embedding_weights = np.random.random((word_len+1, embedding_dim))
    k = 0
    for word, i in word_index.items():
        if i>= NUM_WORDS:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_weights[i] = embedding_vector
            k += 1
    return embedding_weights

# maximum sequence length
def max_seq_len(list1, max_seq_length = 5):
    """Function to find the max sequence length from a list of tokens passed
        list1: list containing the tokens
        max_seq_length: initial length for the variable
    """
    for i in list1:
        if max_seq_length<int((len(i))):
            max_seq_length =len(i)
        else:
            continue
    return max_seq_length



# distince functions
def cosine_distance(left,right):
    """Function to calculate the cosine distance between 2 features
       Args:
           left: feature1
           right: feature2"""
    left = K.l2_normalize(left, axis=-1)
    right = K.l2_normalize(right, axis=-1)
    return -K.mean(left * right, axis=-1, keepdims=True)

    # absolute distance* we are not using this in the code
def exponent_neg_manhattan_distance(left, right):
    """Function to calculate the manhattan distince between 2 features
       Args:
           left: feature 1
           right: feature 2"""
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# functions for relatedness
def cosine_distance(left,right):
    left = K.l2_normalize(left, axis=-1)
    right = K.l2_normalize(right, axis=-1)
    return -K.mean(left * right, axis=-1, keepdims=True)

def pearson_correlation(y_true, y_pred):
    # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    fs_pred = y_pred - K.mean(y_pred)
    fs_true = y_true - K.mean(y_true)
    covariance = K.mean(fs_true * fs_pred)
    
    stdv_true = K.std(y_true)
    stdv_pred = K.std(y_pred)
    
    return covariance / (stdv_true * stdv_pred)
