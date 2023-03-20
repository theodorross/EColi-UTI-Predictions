from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import altair as alt

import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB



def splitData(*arrays, training_frac=0.8, seed=21687):
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
    idx_list = np.arange(len(arrays[0]))
    np.random.seed(seed)
    np.random.shuffle(idx_list)

    ## Split the shuffled index list into training and test subsets with size determined by training_frac
    training_idx = idx_list[ : int(len(idx_list)*training_frac)]
    test_idx = idx_list[int(len(idx_list)*training_frac) : ]

    ## Create training and test arrays
    train_arrs = []
    test_arrs = []
    for arr in arrays:
        train_arrs.append(arr[training_idx,...])
        test_arrs.append(arr[test_idx,...])

    return train_arrs, test_arrs



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

    return classifier_dict


def testPerformance(y_true, y_pred, classifier_name=None, verbose=True):
	'''Test and print out the performance of a classifier
	Arguments
		- y_true (numpy array) : classification labels
		- y_pred (numpy array) : predicted classifications
		- classifier_name (string) : name of the classifier being tested
		- verbose (bool) : controls if the performance metrics are printed out
	
	Returns
		- (float) : raw classification accuracy
		- (float) : precision score
		- (float) : recall score
		- (float) : confusion matrix
	'''
	## Determine how many classes there are
	if len(np.unique(y_true)) == 2: classifier_type = "binary"
	else: classifier_type = "micro"

	## Compute performance metrics
	acc = np.mean(y_true==y_pred)
	conf = sk.metrics.confusion_matrix(y_true, y_pred)
	precision = sk.metrics.precision_score(y_true, y_pred, average=classifier_type, zero_division=True)
	recall = sk.metrics.recall_score(y_true, y_pred, average=classifier_type, zero_division=True)
	sensitivity = conf[1,1]/conf[1,:].sum()
	specificity = conf[0,0]/conf[0,:].sum()

	## Print performance metrics
	out_str = ""
	if classifier_name is not None: out_str += f"\n\n\t{classifier_name} Classifier\n"
	out_str += f"Accuracy: \t{acc:.5f}\n"
	out_str += f"Precision: \t{precision:.5f}\n"
	out_str += f"Recall:  \t{recall:.5f}\n"
	out_str += f"Specificity: \t{specificity:.5f}\n"
	out_str += f"Sensitivity: \t{sensitivity:.5f}\n"
	out_str += f"Confusion Matrix: \n{conf}\n"

	if verbose:
		print(out_str, end="")

	return acc, precision, recall, specificity, sensitivity, conf, out_str



def getYearlyFractions(label, year):
    year_axis = np.unique(year)
    year_masks = [year==yr for yr in year_axis]
    year_frac = [np.sum(label[mask]==1)/mask.sum() for mask in year_masks] 
    return year_frac




def testClassifiers(classifier_dict, **tups):

    ## Initialize performance summary data structures.
    charts = []
    printstr = ""

    ## Compute false positve and false negative rates for the test set.
    fpr_dict = {}
    fnr_dict = {}
    for cname,c in classifier_dict.items():
        x,y,_ = tups["Test"]
        y_pred = c.predict(x)
        _,_,_,_,_,conf,_ = testPerformance(y, y_pred, cname, verbose=False)
        fpr = conf[0,1] / conf[0,:].sum()
        fnr = conf[1,0] / conf[1,:].sum()
        fpr_dict[cname] = fpr
        fnr_dict[cname] = fnr


    ## Compute desired metrics for each dataset.
    for tup_name,tup in tups.items():
        ## Isolate data array and labels.
        x,y,year = tup

        ## Add a header to the summary text.
        printstr += "\n\n" + "="*56 + "\n"
        printstr += "="*21 + f" {tup_name:12s} " +"="*21 + "\n"
        printstr += "="*56

        # Compute the true fraction of samples belonging to 131-C
        year_axis = np.unique(year)
        true_year_frac = getYearlyFractions(y, year)

        # print(true_year_frac)
        ## Intialize a dataframe to plot fraction predictions
        alt_df = pd.DataFrame({"Year":year_axis, "Truth": true_year_frac})

        for cname,c in classifier_dict.items():

            ## Compute the performance metrics for the current dataset-classifier combination
            y_pred = c.predict(x)
            # _,_,_,_,_,conf,outstr = testPerformance(y, y_pred, cname, verbose=False)
            # printstr += outstr
            performance_metrics = testPerformance(y, y_pred, cname, verbose=False)
            printstr += performance_metrics[-1]

            ## Compute predicted fractions of 131-C each year
            pred_year_frac = getYearlyFractions(y_pred, year)
            alt_df[cname] = pred_year_frac

            ## Add predicted information to the output string
            printstr += "\tYear  |  Predicted Fraction\n\t" + "-"*27 + "\n"
            for jx,yr in enumerate(year_axis):
                printstr += f"\t{yr}  | \t {pred_year_frac[jx]:3f}\n"


        ## Impute the colums of the yearly fraction dataframe to produce 
        # long-form data.
        alt_df = alt_df.melt("Year", var_name="Classifier", value_name="Fraction")
        
        ## Broadcast the FPR and FNR values across all predicted yearly fractions.
        fpr_arr = np.zeros(len(alt_df))
        fnr_arr = np.zeros(len(alt_df))
        for cname in classifier_dict.keys():
            c_mask = alt_df["Classifier"].to_numpy() == cname
            fpr_arr[c_mask] = fpr_dict[cname]
            fnr_arr[c_mask] = fnr_dict[cname]

        ## Use the FPR and FNR values to compute expected minimum and maximum 
        # values for yearly fraction predictions.
        alt_df["min"] = alt_df["Fraction"].to_numpy() * (1-fpr_arr)
        alt_df["max"] = alt_df["Fraction"].to_numpy() * (1+fnr_arr)

        ## Visualize the fraction predictions
        line = alt.Chart(alt_df).mark_line().encode(
            x="Year:O",
            y="Fraction:Q",
            color="Classifier:N"
        ).properties(
            title=tup_name
        )
        err_band = alt.Chart(alt_df).mark_area(opacity=0.5).encode(
            x="Year:O",
            y=alt.Y("max:Q", title="Fraction"),
            y2=alt.Y2("min:Q", title="Fraction"),
            color="Classifier:N"
        )
        charts.append(line+err_band)

    ## Combine all the produced visualizations
    chart = alt.hconcat()
    for c in charts:
        chart |= c

    return printstr, chart
