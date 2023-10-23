import numpy as np
import pandas as pd
import altair as alt

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from scipy.stats import pearsonr
from matplotlib import pyplot as plt

def splitData(*arrays, training_frac=0.8, seed=21687):
    '''
    Split data into training and test datasets.

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



def initClassifiers(verbosity=0):
    '''
    Initializes two pre-defined classifiers chosen for performance comparisson.

    Returns
        - (dict) : Dictionary with classifier names for keys corresponding to classifier objects.
        - (dict) : Dictionary with the same keys but boolean values corresponding to whether or not
                    the data should be normalized for the classifier corresponding to the key.
    '''

    ## Initialize the Random Forest classifier for cross-validation
    rf = RandomForestClassifier(class_weight="balanced")
    rf_grid = {"n_estimators":[10,50,100,250,500,1000],
               "class_weight":["balanced",None]}
    rf_cv = GridSearchCV(estimator=rf, param_grid=rf_grid, scoring="f1", 
                         cv=5, verbose=verbosity)
    
    ## Initialize the XGBoost classifier for cross-validation
    xgb = XGBClassifier(eval_metric="mlogloss")
    xgb_grid = {"n_estimators":[10,50,100,250],
                "learning_rate":[0.01,0.1,0.5,1.0,1.5],
                "max_depth":[2,4,6,10,15]}
    xgb_cv = GridSearchCV(estimator=xgb, param_grid=xgb_grid, scoring="f1", 
                          cv=5, verbose=verbosity)

    ## Returnn the classfiers
    classifier_dict = {"Random Forest": rf_cv, "XGBoost": xgb_cv}

    return classifier_dict


def testPerformance(y_true:np.ndarray, y_pred:np.ndarray, classifier_name:str=None, verbose:bool=True):
    '''
    Test and print out the performance of a classifier

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
    f1 = sk.metrics.f1_score(y_true, y_pred)
    conf = sk.metrics.confusion_matrix(y_true, y_pred)
    precision = sk.metrics.precision_score(y_true, y_pred, average=classifier_type, zero_division=True)
    recall = sk.metrics.recall_score(y_true, y_pred, average=classifier_type, zero_division=True)
    sensitivity = conf[1,1]/conf[1,:].sum()
    specificity = conf[0,0]/conf[0,:].sum()

    ## Print performance metrics
    out_str = ""
    if classifier_name is not None: out_str += f"\n\n\t{classifier_name} Classifier\n"
    out_str += f"Accuracy: \t{acc:.5f}\n"
    out_str += f"F1-Score: \t{f1:.5f}\n"
    out_str += f"Precision: \t{precision:.5f}\n"
    out_str += f"Recall:  \t{recall:.5f}\n"
    out_str += f"Specificity: \t{specificity:.5f}\n"
    out_str += f"Sensitivity: \t{sensitivity:.5f}\n"
    out_str += f"Confusion Matrix: \n{conf}\n"

    if verbose:
        print(out_str, end="")

    return acc, precision, recall, specificity, sensitivity, conf, out_str



def getYearlyFractions(label:np.ndarray, year:np.ndarray):
    '''
    Helper function to compute the fraction of isolates that are positively labeled
    in each year.

    Arguments  
        - label (numpy array) : vector of binary labels.
        - year (numpy array) : vector of year data.
    
    Returns
        -  (numpy array) : vector of fractions for each unique year in the 'year' vector.
    '''
    year_axis = np.unique(year)
    year_masks = [year==yr for yr in year_axis]
    year_frac = [np.sum(label[mask]==1)/mask.sum() for mask in year_masks] 
    return year_frac


def correlationPlots(savepath, **kwargs):
    '''
    Function to generate a correlation plot for 

    Arguments:
        - savepath (str) : path to save the image to, without a file extension.
        - **kwargs (array-likes) : keyword-arguments of the names of each fraction
                                    array and the arrays themselves. Must include 
                                    a "Truth" keyword array.

    '''

    ## Define the dataframe for generating the plots
    df = pd.DataFrame(kwargs)
    df = df.melt("Truth", var_name="Classifier", value_name="Prediction")
    
    ## Plot the fraction data
    chart = alt.Chart(df).mark_point().encode(
        x = "Truth:Q",
        y = "Prediction:Q",
        color = "Classifier:N",
        shape = "Classifier:N"
    )
    line = alt.Chart().mark_rule().encode(
        x = alt.datum(0),
        x2 = alt.datum(0.25),
        y = alt.datum(0),
        y2 = alt.datum(0.25)
    )
    chart = line + chart

    ## Save the plots
    chart.save(f"{savepath}.png")
    chart.save(f"{savepath}.svg")



def testClassifiers(classifier_dict:dict, **tups:tuple):
    '''
    Function to test the performance of input classifiers.  Computes various metrics 
    including accuracy, F1-score, precision, recall, sensitivity, and specificity.  
    The false-positive and -negative rates are also computed for computing error bands
    in the yearly fraction plots.  These yearly fraction plots are also computed 
    independently for the test and training sets.

    Arguments
        - classifier_dict (dict) : dictionary containing the classifier objects to be 
                                    evaluated.  The keys of the dictionary will be used
                                    as the classifier names.
        - tups (tuples) : any amount of data tuples containing (features, labels, year) 
                            for each sample. The keywords given for each tuple will be 
                            used as the label for that dataset.

    Returns
        - (str) : string with summary statistics formatted for each input classifier and
                    dataset.
        - (altair chart) : altair chart visualizing the yearly prevalence in all input 
                            datasets.
        - (dict) : dictionary containing the false positive rates of the input classifiers.
        - (dict) : dictionary containing the false negative rates of the input classifiers.
    '''
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

        # Initialize a dictionary for generating correlation plots
        correlation_dict = {"Truth": true_year_frac}

        ## Intialize a dataframe to plot fraction predictions
        alt_df = pd.DataFrame({"Year":year_axis, "Truth": true_year_frac})

        for cname,c in classifier_dict.items():

            ## Compute the performance metrics for the current dataset-classifier combination
            y_pred = c.predict(x)
            performance_metrics = testPerformance(y, y_pred, cname, verbose=False)
            printstr += performance_metrics[-1]

            ## Compute predicted fractions of 131-C each year
            pred_year_frac = getYearlyFractions(y_pred, year)
            alt_df[cname] = pred_year_frac
            correlation_dict[cname] = pred_year_frac

            ## Add predicted information to the output string
            printstr += "\tYear  |  Predicted Fraction\n\t" + "-"*27 + "\n"
            for jx,yr in enumerate(year_axis):
                printstr += f"\t{yr}  | \t {pred_year_frac[jx]:3f}\n"

            ## Compute the Pearson correlation coeffiecient between the truth and 
            #  predicted fraction values
            r_val = pearsonr(pred_year_frac, true_year_frac)
            printstr += f"\nPearson R: {r_val.statistic:5f}\nR P-value: {r_val.pvalue:5f}\n"

        ## Create and save correlation plots
        correlationPlots(f"output/correlation_{tup_name}", **correlation_dict)


        ## Impute the colums of the yearly fraction dataframe to produce 
        #  long-form data.
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
        chrt = line+err_band
        chrt = chrt.properties(width=300, height=300)
        charts.append(chrt)

    ## Combine all the produced visualizations
    output_chart = alt.hconcat()
    for c in charts:
        output_chart |= c
    output_chart = output_chart.configure_axis(
        labelFontSize=18,
        titleFontSize=18,
        labelLimit=400
    ).configure_legend(
        labelFontSize=18,
        titleFontSize=20,
        titleLimit=400,
        labelLimit=600
    ).configure_header(
        labelFontSize=20
    ).configure_title(
        fontSize=22
    )

    # print(type(output_chart))
    return printstr, output_chart, fpr_dict, fnr_dict



def predictClassifiers(classifier_dict:dict, x:np.ndarray, year:np.ndarray, 
                       fpr_dict:dict, fnr_dict:dict, uti_idx=None):
    '''
    Use the trained classifiers to predict and analyze an unlabeled dataset.  This
    function also saves the raw predictions to "output/uti-predictions.csv".

    Arguments
        - classifier_dict (dict) : dictionary of trained classifiers.
        - x (numpy array) : data array with the first dimension corresponding to
                            the number of datapoints.
        - year (numpy array) : year metadata value for each input isolate.
        - fpr_dict (dict) : dictionary of false-postive rates for the classifiers,
                            returned by 'testClassifiers()'.
        - fnr_dict (dict) : dictionary of false-negative rates for the classifiers,
                            returned by 'testClassifiers()'.
        - uti_idx (numpy array, list, or pandas index) : index column for the uti data.

    Returns
        - (altair chart) : visualization of the fraction of positive predictions in 
                            each year.
    '''    

    alt_df = pd.DataFrame({"Year":np.unique(year)})
    preds = {"Year":year}
    
    ## Use each classifier to make predictions
    for cname,c in classifier_dict.items():

        ## Use the trained classifier to predict the membership in ST-131 Clade C for all
        # isolates
        y = c.predict(x)
        preds[cname] = y

        ## Compute the fraction of positive predictions for each year
        year_fracs = getYearlyFractions(y, year)
        alt_df[cname] = year_fracs

    ## Impute the colums of the yearly fraction dataframe to produce 
    # long-form data.
    alt_df = alt_df.melt("Year", var_name="Classifier", value_name="Fraction")

    ## Compute the estimated error band using false-positive and -negative rates.
    for ix in alt_df.index:
        cname = alt_df.loc[ix, "Classifier"]
        fpr, fnr = fpr_dict[cname], fnr_dict[cname]
        alt_df.loc[ix,"min"] = alt_df.loc[ix,"Fraction"] * (1-fpr)
        alt_df.loc[ix,"max"] = alt_df.loc[ix,"Fraction"] * (1+fnr)


    
    ## Visualize the predicted yearly fractions
    line = alt.Chart(alt_df).mark_line().encode(
        x="Year:O",
        y="Fraction:Q",
        color="Classifier"
    )
    err = alt.Chart(alt_df).mark_area(opacity=0.5).encode(
        x="Year:O",
        y=alt.Y("max:Q", title="Fraction"),
        y2=alt.Y2("min:Q", title="Fraction"),
        color="Classifier:N"
    )
    chart = line+err
    chart = chart.properties(
        width=600, 
        height=300
    ).configure_axis(
        labelFontSize=18,
        titleFontSize=18,
        labelLimit=400
    ).configure_legend(
        labelFontSize=18,
        titleFontSize=20,
        titleLimit=400,
        labelLimit=600
    ).configure_header(
        labelFontSize=20
    ).configure_title(
        fontSize=22
    )

    ## Collate and summarize the predictions
    pred_df = pd.DataFrame(preds, index=uti_idx)
    pred_df.index.name = "Row Number"
    pred_df.to_csv("output/uti-predictions.csv")

    return chart