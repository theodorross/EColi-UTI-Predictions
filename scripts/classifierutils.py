from xgboost import XGBClassifier
import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB



def splitData(x,y, training_frac=0.8, seed=21687):
    '''
    Split data into training and test datasets
    Arguments
        - x (numpy array) : data samples, with the first index corresponding to individual samples.
        - y (numpy array) : data labels.
        - training_frac (float in [0,1]) : fraction of the data that will be kept in the training set.
        - seed (float or int) : random seed used for the index shuffling operation, kept constant to maintain 
                                constant datsets between tests.
    Returns
        - (numpy array) : training dataset
        - (numpy array) : normalized training dataset
        - (numpy array) : training label set
        - (numpy array) : test dataset
        - (numpy array) : normalized test dataset
        - (numpy array) : test label set
    '''
    ## Create a list of indices and randomly shuffle them
    idx_list = np.arange(len(y))
    np.random.seed(seed)
    np.random.shuffle(idx_list)

    ## Split the shuffled index list into training and test subsets with size determined by training_frac
    training_idx = idx_list[ : int(len(idx_list)*training_frac)]
    test_idx = idx_list[int(len(idx_list)*training_frac) : ]

    ## Normalize data
    denom = np.abs(x).max()
    x_norm = x/denom

    ## Create training and test data and label sets using the split index arrays
    xtrain,ytrain = x[training_idx,...], y[training_idx]
    xtrain_norm = x_norm[training_idx,...]
    xtest,ytest = x[test_idx,...], y[test_idx]
    xtest_norm = x_norm[test_idx,...]

    return xtrain,xtrain_norm,ytrain, xtest,xtest_norm,ytest



def initClassifiers():
    '''
    Initializes two pre-defined classifiers chosen for performance comparisson.

    Returns
        - (dict) : Dictionary with classifier names for keys corresponding to classifier objects.
        - (dict) : Dictionary with the same keys but boolean values corresponding to whether or not
                    the data should be normalized for the classifier corresponding to the key.
    '''

    ## Initialize classifiers
    classifier_dict = {"Random Forest": RandomForestClassifier(class_weight="balanced"),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")}
    # normalize_dict = {"Random Forest": False,
    #                     "XGBoost": False}

    return classifier_dict



def testClassifiers(classifier_dict, **dfs):

    charts = []
    printstr = ""
    metrics = {}

    for df_name,df in dfs.items():

        ## Add a header to the summary text
        printstr += "\n\n" + "="*56 + "\n"
        printstr += "="*21 + f" {df_name:12s} " +"="*21 + "\n"
        printstr += "="*56

        ## Isolate data and labels
        print(df)


        pass

    return
