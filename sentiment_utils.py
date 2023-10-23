
# YOUR NAMES HERE: Julie Geller(4120), Shae Marks(4120)


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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import math
import statistics as stats



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
NOTE: for all of the following functions, we have provided the function signature and docstring, *that we used*, as a guide.
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
    precision = precision_score(dev_y, preds)
    recall = recall_score(dev_y, preds)
    f1 = f1_score(dev_y, preds)
    accuracy = accuracy_score(dev_y, preds)
    if verbose:
        print(precision, recall, f1, accuracy)
    return (precision, recall, f1, accuracy)


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
    # save training and dev features and labels
    X_train = [tup[0] for tup in train_feats]
    y_train = [tup[1] for tup in train_feats]
    X_dev = [tup[0] for tup in dev_feats]
    y_dev = [tup[1] for tup in dev_feats]
    
    # intialize y's for graph
    y1 = [] # precision
    y2 = [] # recall
    y3 = [] # f1 score
    y4 = [] # accuracy

    # create percent split of training data to use
    percent_train_data = [i for i in np.arange(0.2, 1.2, 0.2)]
    # split training data into above percent splits
    X_train_splits = [X_train[0:math.ceil(len(X_train)*percent)] for percent in percent_train_data]
    y_train_splits = [y_train[0:math.ceil(len(X_train)*percent)] for percent in percent_train_data]
    # intialize which percent split to use
    i=0
    # get metrics for each percent split and save to corresponding graph y list
    for x_split_train, y_split_train in zip(X_train_splits, y_train_splits):
        if verbose:
            print(percent_train_data[i], 'of training data used')
        # get model metrics
        metrics = metrics_fun(x_split_train, y_split_train, X_dev, y_dev, verbose=verbose)
        # save y's to appropriate graph y list
        y1.append(metrics[0]) 
        y2.append(metrics[1])
        y3.append(metrics[2])
        y4.append(metrics[3])
        # update percent split index to use
        i+=1
    
    # plot graph
    plt.plot(percent_train_data, y1, label = "precision") 
    plt.plot(percent_train_data, y2, label = "recall") 
    plt.plot(percent_train_data, y3, label = "f1 score") 
    plt.plot(percent_train_data, y4, label = "accuracy") 

    plt.title(kind+' Model Performance Analysis')
    plt.xlabel('Percent of Training Data Used')
    plt.ylabel('Metric score')
    plt.legend() 
    plt.show()
    if savepath is not None:
        plt.savefig(savepath+'.png')


def log_reg_metrics(X_train: list, y_train: list, X_dev: list, y_dev: list, verbose: bool=False):
    """
    Generates performance metrics for a Logistic Regression model trained onthe given training data
     and tested on the given dev data.
    Args:
        X_train: list of list of int (featurized training data)
        y_train: list of int (trianing data labels)
        y_dev: list of int (dev data labels)
        verbose: bool (if model metrics should be printed)
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    model = LogisticRegression(max_iter=100000)
    model.fit(X_train, y_train)
    preds = model.predict(X_dev)
    return get_prfa(y_dev,  preds, verbose=verbose)


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


def featurize_own(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
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
    # initialize count of datapoints already featurized 
    count = 0
    total_datapoints = len(data_to_be_featurized_X[0])
    
    if verbose:
        print('Number of datapoints featurized')
    # initialize X matrix 
    X = []
    # featurize each datapoint 
    for review in data_to_be_featurized_X:
        # get the count of each word in the vocab in the current review 
        rev_counter = Counter(review)
        x = [rev_counter[word] for word in vocab]
        if binary:
            # transform counts above 0 to be 1 if doing binary featurization
            x = [1 if c>0 else 0 for c in x]
        # add x vector matrix X 
        X.append(x)
        # increase count of datapoints featurized already
        count += 1
        # print progress is verbose is required
        if verbose:
            print(count, '/', total_datapoints)
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


def featurize(type: str, vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
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
        X = featurize_own(vocab, data_to_be_featurized_X, binary, verbose)
    elif type == 'CV':
        X = featurize_CV(vocab, data_to_be_featurized_X, binary) 
    else:
        raise Exception('Invalid featurization type provided. Must be either "own" or "CV".')
    return X


def percent_zeros_help(vect):
     """
    Helper function for percent_zeros that calculates the percent of entries that are 0 in the given vector.
    Args:
        vect: list of int (single row of featurized data)
    Returns:
        float
  """
     zeros = [i for i in vect if i==0]
     pct = len(zeros)/len(vect)
     return pct


def percent_zeros(X):
    """
    Calculate the average of the percent of 0's in each given vector.
    Args:
        X: list of list of int (featurized data)
    Returns:
        float
  """
    pcts = [percent_zeros_help(x) for x in X]
    return stats.mean(pcts)


def create_log_reg(X_train: list, y_train: list) -> list:
    """
    Creates a logistic regression model based on the given training data.
    Args:
        X_train: str(either 'own' or 'CV')
        y_train: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        model
  """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model