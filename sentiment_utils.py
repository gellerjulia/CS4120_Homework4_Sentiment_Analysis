
# YOUR NAMES HERE: Julie Geller, Shae Marks


"""
Felix Muzny
CS 4/6120
Homework 4
Fall 2023

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import defaultdict, Counter
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
# for plotting
import matplotlib.pyplot as plt
# so that we can indicate a function in a type hint
from typing import Callable
nltk.download('punkt')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def generate_tuples_from_file(training_file_path: str) -> list:
    """
    Generates data from file formated like:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]
    
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of lists of tokens and a list of int labels
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split("\t")
        if len(dataInReview) != 3:
            continue
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            X.append(nltk.word_tokenize(t[1]))
            y.append(int(t[2]))
    f.close()  
    return X, y


"""
NOTE: for all of the following functions, we have prodived the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""


def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    #TODO: implement this function
    pass

def create_training_graph(metrics_fun: Callable, train_feats: list, dev_feats: list, kind: str, savepath: str = None, verbose: bool = False) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data and dev data and returns a tuple of metrics
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    #TODO: implement this function
    pass



def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    vocab = []
    for row in all_train_data_X:
        [vocab.append(wrd) for wrd in row]
    vocab = list(set(vocab))
    return vocab

def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    # initalize X matrix 
    cnt = 0
    if verbose:
        print('Number of datapoints featurized')
    X = []
    for row in data_to_be_featurized_X:
        x = [row.count(word) for word in vocab]
        if binary:
            x = [1 if c>0 else 0 for c in x]
        # add x values to corresponding X matrix
        X.append(x)
        cnt += 1
        if verbose:
            print(cnt+'/'+len(data_to_be_featurized_X[0]))
    return X

def featurize_CV(vocab: list, data_to_be_featurized_X: list, binary: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data using CountVectorizer.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    vectorizer = CountVectorizer(vocabulary=vocab, binary=binary)
    X = vectorizer.fit_transform(data_to_be_featurized_X)
    # turn X into a list of lists for standard vector representation
    X = X.toarray().tolist()
    return X

def get_featurized(type: str, vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data using own vectorization function or CountVectorizer.
    Args:
        type: str(either 'own' or 'CV')
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    if type == 'own':
        X = featurize(vocab, data_to_be_featurized_X, binary, verbose)
    elif type == 'CV':
        X = featurize_CV(vocab, data_to_be_featurized_X, binary) 
    else:
        raise Exception('Invalid featurizatino type provided. Must be either "own" or "CV".')
    return X
  