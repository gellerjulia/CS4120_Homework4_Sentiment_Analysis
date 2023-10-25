
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
from nltk.classify import NaiveBayesClassifier
import math
import statistics as stats
from keras.models import Sequential
from keras.layers import Dense



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
        print("Precision:", precision, " Recall:", recall, " f1:", f1, " Accuracy:", accuracy)
    return (precision, recall, f1, accuracy)


def create_training_graph(metrics_fun: Callable, train_feats: list, dev_feats: list, kind: str, savepath: str = None, verbose: bool = False, num_epochs=None, neural_net_verbose: bool = True) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data and dev data and returns a tuple of metrics
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
        num_epochs: int (number of epochs to sue if model is a neural network)
        neural_net_verbose: whether or not to print epoch progress and accuracy when training a neural network
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
    percent_train_data = [i for i in np.arange(0.1, 1.1, 0.1)] # [0.1, 0.2, ..., 1.0]
    # split training data into above percent splits
    X_train_splits = [X_train[0:math.ceil(len(X_train)*percent)] for percent in percent_train_data]
    y_train_splits = [y_train[0:math.ceil(len(X_train)*percent)] for percent in percent_train_data]
    # intialize which percent split to use
    i=0
    # get metrics for each percent split and save to corresponding graph y list
    for x_split_train, y_split_train in zip(X_train_splits, y_train_splits):
        if verbose:
            print("\n", percent_train_data[i], 'of training data used')
        # get model metrics
        if num_epochs is None:
            metrics = metrics_fun(x_split_train, y_split_train, X_dev, y_dev, verbose=verbose)
        else:
            metrics = metrics_fun(x_split_train, y_split_train, X_dev, y_dev, num_epochs, verbose=verbose, neural_net_verbose=neural_net_verbose)

        # save y's to appropriate graph y list
        y1.append(metrics[0]) 
        y2.append(metrics[1])
        y3.append(metrics[2])
        y4.append(metrics[3])
        # update percent split index to use
        i+=1
    
    # plot graph
    # format x-axis in percent form: [10%, 20%, ... 100%]
    x_axis_percents = [str(int(percent*100)) + "%" for percent in percent_train_data]
    plt.plot(x_axis_percents, y1, label = "precision") 
    plt.plot(x_axis_percents, y2, label = "recall") 
    plt.plot(x_axis_percents, y3, label = "f1 score") 
    plt.plot(x_axis_percents, y4, label = "accuracy") 
    
    plt.title(kind+' Model Performance Analysis')
    plt.xlabel('Percent of Training Data Used')
    plt.ylabel('Metric score')
    plt.legend() 
    plt.xticks(x_axis_percents)

    if savepath is not None:
        plt.savefig(savepath)
        
    plt.show()


def log_reg_metrics(X_train: list, y_train: list, X_dev: list, y_dev: list, verbose: bool=False):
    """
    Generates performance metrics for a Logistic Regression model trained on the given training data
     and tested on the given dev data.
    Args:
        X_train: list of list of int (featurized training data)
        y_train: list of int (training data labels)
        y_dev: list of int (dev data labels)
        verbose: bool (if model metrics should be printed)
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    model = LogisticRegression(max_iter=100000)
    model.fit(X_train, y_train)
    preds = model.predict(X_dev)
    return get_prfa(y_dev,  preds, verbose=verbose)


def neural_net_metrics(X_train: list, y_train: list, X_dev: list, y_dev: list, num_epochs: int, verbose: bool=False, neural_net_verbose: bool=True):
    """
    Generates performance metrics for a Neural Network model trained on the given training data
     and tested on the given dev data. Neural Network uses 1 hidden layer with 100 hidden units.
    Args:
        num_epochs: int (number of training epochs)
        X_train: list of list of int (featurized training data)
        y_train: list of int (training data labels)
        X_dev: list of list of int (featurized dev data)
        y_dev: list of int (dev data labels)
        input_dim: int (number of dimensions in input)
        verbose: bool (if model metrics should be printed)
        neural_net_verbose: bool (if epoch training progress should be printed)
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    # make sure all lists are numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_dev = np.array(X_dev)

    # define model parameters
    hidden_units = 100
    input_dim = X_train.shape[1]

    # instantiate model
    model = Sequential()

    # hidden layer 
    model.add(Dense(units=hidden_units, activation='relu', input_dim=input_dim))
    # output layer
    model.add(Dense(units=1, activation='sigmoid'))

    # configure the learning process
    model.compile(loss='binary_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    
    # train model 
    model.fit(X_train, y_train, epochs=num_epochs, verbose=neural_net_verbose)

    # get model predictions
    preds = model.predict(X_dev, verbose=neural_net_verbose)
    # make classification decision based on 0.5 as threshold
    preds = [1 if y >= 0.5 else 0 for y in preds]
    return get_prfa(y_dev, preds, verbose=verbose)
    
    
def naive_bayes_metrics(X_train: list, y_train: list, X_dev: list, y_dev: list, verbose: bool=False):
    """
    Generates performance metrics for a Naive Bayes model trained on the given training data
     and tested on the given dev data.
    Args:
        X_train: list of list of int (featurized training data)
        y_train: list of int (training data labels)
        X_dev: list of list of int (featurized dev data)
        y_dev: list of int (dev data labels)
        verbose: bool (if model metrics should be printed)
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    train_data = [(x, y) for x,y in zip(X_train, y_train)]
    model = NaiveBayesClassifier.train(train_data)
    preds = [model.classify(sample) for sample in X_dev]
    return get_prfa(y_dev, preds, verbose=verbose)


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
        # increment count of datapoints featurized already
        count += 1
        # print progress if verbose 
        if verbose:
            print(count, '/', total_datapoints)

    return X


def featurize_CV(train_data_to_be_featurized_X: list, dev_data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> tuple:
    """
    Create vectorized BoW representations of the given data using CountVectorizer.
    Args:
        train_data_to_be_featurized_X: a list of training data to be featurized in the format [[word1, word2, ...], ...]
        dev_data_to_be_featurized_X: a list of dev data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: whether or not to print out learned vocab size
    Returns:
        a tuple where each element is a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    vectorizer = CountVectorizer(binary=binary)
    vectorizer = vectorizer.fit(train_data_to_be_featurized_X)
    X_train = vectorizer.transform(train_data_to_be_featurized_X)
    X_dev = vectorizer.transform(dev_data_to_be_featurized_X)
    # turn X's into a list of lists for standard vector representation
    X_train = X_train.toarray().tolist()
    X_dev = X_dev.toarray().tolist()
    
    if verbose:
        print("Vocab size:", len(vectorizer.vocabulary_))

    return X_train, X_dev


def featurize(type: str, train_data_to_be_featurized_X: list, dev_data_to_be_featurized_X: list, vocab: list = [], binary: bool = False, verbose: bool = False) -> tuple:
    """
    Create vectorized BoW representations of the given data using own vectorization function or CountVectorizer.
    Args:
        type: str(either 'own' or 'CV')
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out additional information
    Returns:
        a tuple where each element is a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    if type == 'own':
        X_train = featurize_own(vocab, train_data_to_be_featurized_X, binary, verbose)
        X_dev = featurize_own(vocab, dev_data_to_be_featurized_X, binary, verbose)
    elif type == 'CV':
        X_train, X_dev = featurize_CV(train_data_to_be_featurized_X, dev_data_to_be_featurized_X, binary, verbose) 
    else:
        raise Exception('Invalid featurization type provided. Must be either "own" or "CV".')
    return X_train, X_dev

def naive_bayes_featurize(data_to_be_featurized_X: list, vocab: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create BoW representations for a list of samples to be used by NLTK's NaiveBayesClassifier.
    Args:
        data_to_be_featurized_X: list of of lists, where each inner list is the words from a tokenized data sample 
        vocab: a list of words in the vocabulary
        binary: whether or not to use binary features
        verbose: whether or not to print additional information about the data being processed 
    Returns:
        a list of dictionaries (one for each sample) representing the words present in both the vocab and the sample
        - for binary representations, in the format {word1: True, word2: True, ...}
        - for multinomial representations, in the format {word1: count1, word2: count2, ...}
    """ 
    featurized_data = []    
    for i in range(len(data_to_be_featurized_X)):
        featurized_sample = naive_bayes_word_feats(data_to_be_featurized_X[i], vocab, binary=binary, verbose=verbose)
        featurized_data.append(featurized_sample)

    return featurized_data


def naive_bayes_word_feats(doc_words: list, vocab: list, binary: bool = False, verbose: bool = False) -> dict:   
    """
    Create BoW representations of the given data to be used by NLTK's NaiveBayesClassifier.
    Args:
        doc_words: list of words from a tokenized data sample 
        vocab: a list of words in the vocabulary
        binary: whether or not to use binary features
        verbose: whether or not to print additional information about the data being processed 
    Returns:
        a dictionary representing the words present in both the vocab and doc_words 
        - for binary representations, in the format {word1: True, word2: True, ...}
        - for multinomial representations, in the format {word1: count1, word2: count2, ...}
    """ 
    # STUDENTS IMPLEMENT
    doc_counter = Counter(doc_words)

    # for efficiency, only iterate through words that we know are both in the vocab and in doc_words 
    overlap = set.intersection(set(doc_counter.keys()), set(vocab))

    # initialize empty dictionary of features 
    bow_feats = {}
    for word in overlap:
        if binary:
            bow_feats[word] = True
        else:
            bow_feats[word] = doc_counter[word]

    if verbose:
        print("Size of doc_words data:", len(doc_words))
        print("Size of vocab:", len(vocab))
        print("Size of overlap between doc_words and vocab:", len(overlap))
        print("Feature examples:", list(bow_feats.items())[:3])
        print()
           
    return bow_feats 


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


def create_log_reg(X_train: list, y_train: list) -> LogisticRegression:
    """
    Creates a logistic regression model based on the given training data.
    Args:
        X_train: str(either 'own' or 'CV')
        y_train: a list of words in the vocabulary
    Returns:
        LogisticRegression model
  """
    
    model = LogisticRegression(max_iter=500) # increasing maximum number of iterations to allow model to converge
    model.fit(X_train, y_train)
    return model

