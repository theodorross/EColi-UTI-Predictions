import numpy as np
import pandas as pd
import altair as alt
import os
import pickle

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from scipy.stats import pearsonr, MonteCarloMethod
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



def testClassifiers(df:pd.core.frame.DataFrame, err_df:pd.core.frame.DataFrame, pan_str:str, prefix:str, write_files:bool):
    '''
    Function to test the performance of input classifiers.  Computes various metrics 
    including accuracy, F1-score, precision, recall, sensitivity, and specificity.  
    The false-positive and -negative rates are also computed for computing error bands
    in the yearly fraction plots.  These yearly fraction plots are also computed 
    independently for the test and training sets.  This fucntion also saves the raw
    prediction datsets to "output/{dataset name}-predictions.csv

    Arguments
        - df (DataFrame) : pandas DataFrame containing the predictions on the Test and
                            Training datasets. Must contain columns ["Split","Label",
                            "Year"] as well as any model names used for prediction.
        - classifier_names (list) : list of classifier names to be tested. Must correspond
                            with the names of the model prediction columns in df.
        - pan_str (str) : string indicating whether or not pan-susceptible isolates have
                            been removed.
        - prefix (str) : string for the prefix of file names saved by this function.
        - write_files (bool) : boolean argument controlling whether or not to write the
                            outputs to files. Writes images and model performance metrics
                            when write_files=True.

    Returns
        - (str) : string with summary statistics formatted for each input classifier and
                    dataset.
        - (altair chart) : altair chart visualizing the yearly prevalence in all input 
                            datasets.
        - (altair chart) : altair chart visualizing the correlations with yearly prevelance
                            prevalence predictions and ground truth values.
        - (dict) : dictionary containing the false positive rates of the input classifiers.
        - (dict) : dictionary containing the false negative rates of the input classifiers.
    '''
    ## Initialize performance summary data structures.
    charts = []
    printstr = ""

    # ## Reset the index of err_df for later use
    # err_df = err_df.set_index("Year")

    ## Isolate the classifier names
    classifier_names = df.columns.tolist()[6:]

    ## Split the dataframe into test and training splits
    df_test = df[df["Split"] == "Test"]
    df_train = df[df["Split"] == "Training"]

    ## Initialize a dataframe for plotting model correlations.
    corr_df = pd.DataFrame()

    ## Define the columns to use for each plot
    corr_cols = ["Truth"] + classifier_names + ["Dataset"]
    trend_cols = ["Year","Truth"] + classifier_names

    ## Compute desired metrics for each dataset.
    for split,split_df in {"Test":df_test,"Training":df_train}.items():
        ## Isolate data array and labels.
        # x,y,year,name = tup
        y = split_df["Label"].to_numpy()
        name = split_df.index.to_numpy()
        year = split_df["Year"].to_numpy()
        # x = 

        ## Add a header to the summary text.
        printstr += "\n\n" + "="*56 + "\n"
        printstr += "="*21 + f" {split:12s} " +"="*21 + "\n"
        printstr += "="*56

        # Compute the true fraction of samples belonging to 131-C
        year_axis = np.unique(year)
        true_year_frac = getYearlyFractions(y, year)
        # corr_dict = {"Truth":true_year_frac}

        ## Initialize a dataframe to plot fraction predictions
        alt_df = pd.DataFrame({"Year":year_axis, "Truth": true_year_frac})

        ## Initialize a dataframe for the raw predictions
        pred_df = pd.DataFrame({"Label":y, "Year":year}, index=name)

        for cname in classifier_names:

            ## Extract the model predictions
            y_pred = split_df[cname].to_numpy()

            ## Compute the performance metrics for the current dataset-classifier combination
            # y_pred = c.predict(x)
            performance_metrics = testPerformance(y, y_pred, cname, verbose=False)
            printstr += performance_metrics[-1]

            # ## Add the current classifier's predictions to the dataset's prediction table
            # pred_df[cname] = y_pred

            ## Compute predicted fractions of 131-C each year
            pred_year_frac = getYearlyFractions(y_pred, year)
            alt_df[cname] = pred_year_frac
            # corr_dict[cname] = pred_year_frac

            ## Add predicted information to the output string
            printstr += "\n\tYear  |  Predicted Fraction | True Fraction\n\t" + "-"*43 + "\n"
            for jx,yr in enumerate(year_axis):
                printstr += f"\t{yr}  |\t{pred_year_frac[jx]:3f} \t| \t {true_year_frac[jx]:3f}\n"

            ## Compute the Pearson correlation coeffiecient between the truth and 
            #  predicted fraction values
            r_val = pearsonr(pred_year_frac, true_year_frac, alternative="greater", method=MonteCarloMethod(n_resamples=5e4))
            printstr += f"\nPearson R: {r_val.statistic:5f}\nR P-value: {r_val.pvalue}\n"

        # ## Save the current dataset's predictions
        # pred_df.to_csv(f"training-metrics/{pan_str}/{prefix}_{split}-predictions.csv")

        alt_df["Dataset"] = split


        ## Save correlation plot data
        corr_df = pd.concat([corr_df, alt_df[corr_cols]])

        ## Impute the colums of the yearly fraction dataframe to produce 
        #  long-form data, then convert fractions into percentage.
        alt_df = alt_df[trend_cols].melt("Year", var_name="Classifier", value_name="Fraction")
        alt_df["Fraction"] = alt_df["Fraction"] * 100
        
        ## Use the FPR and FNR values to approximate error margins
        alt_df.set_index(["Year","Classifier"], inplace=True)
        for cname in classifier_names:
            fpr_vec = err_df[f"{cname} FPR"].to_numpy()
            fnr_vec = err_df[f"{cname} FNR"].to_numpy()

            idx_tup = (err_df.index, cname)
            alt_df.loc[idx_tup,"min"] = alt_df.loc[idx_tup,"Fraction"] * (1-fpr_vec)
            alt_df.loc[idx_tup,"max"] = alt_df.loc[idx_tup,"Fraction"] * (1+fnr_vec)

        # Undo the indexing
        alt_df.reset_index(inplace=True)

        ## Visualize the fraction predictions
        line = alt.Chart(alt_df).mark_line().encode(
            x="Year:O",
            y=alt.Y("Fraction:Q", title="Proportion of Isolates (%)"),
            color="Classifier:N"
        ).properties(
            title=split
        )
        err_band = alt.Chart(alt_df).mark_area(opacity=0.5).encode(
            x="Year:O",
            y=alt.Y("max:Q", title="Proportion of Isolates (%)"),
            y2=alt.Y2("min:Q", title="Proportion of Isolates (%)"),
            color="Classifier:N"
        )
        chrt = line+err_band
        chrt = chrt.properties(width=300, height=300)
        charts.append(chrt)

    ## Produce the correlation plots
    corr_df = corr_df.melt(["Truth","Dataset"], var_name="Classifier", value_name="Prediction")
    max_corner = max(corr_df["Truth"].max(), corr_df["Prediction"].max())
    corr_plot = alt.Chart(corr_df).mark_point().encode(
        x = alt.X("Truth:Q").scale(domain=[0,max_corner]),
        y = alt.Y("Prediction:Q").scale(domain=[0,max_corner]),
        shape = "Classifier:N",
        color = "Classifier:N"
    )
    corr_line = alt.Chart(corr_df).mark_rule(clip=True).encode(
        x = alt.datum(0),
        x2 = alt.datum(1),
        y = alt.datum(0),
        y2 = alt.datum(1)
    )
    corr_chart = alt.layer(
        corr_plot, 
        corr_line,
        data=corr_df
    ).facet(
        column=alt.Column("Dataset:N", title="")
    ).resolve_axis(
        x='independent',
        y='independent',
    )

    ## Combine all the produced visualizations
    output_chart = alt.hconcat()
    for c in charts:
        output_chart |= c
    output_chart = output_chart

    ## Write the outputs to files if specified
    if write_files:
        # Write the text output
        with open(f"output/{pan_str}/{prefix}.classifier_metrics.txt","w") as f:
            f.write(printstr)

        # Save the yearly trend chart
        output_chart.save(f"output/{pan_str}/{prefix}.yearly_fractions.png")

        # Save the yearly trend correlation chart
        corr_chart.save(f"output/{pan_str}/{prefix}.correlations.png")
        
    # return printstr, output_chart, corr_chart, fpr_dict, fnr_dict
    return printstr, output_chart, corr_chart



def predictClassifiers(df:pd.core.frame.DataFrame, err_df:pd.core.frame.DataFrame, pan_str:str, prefix:str):
    '''
    Use the trained classifiers to predict and analyze an unlabeled dataset.  This
    function also saves the raw predictions to "output/uti-predictions.csv".

    Arguments
        - df (DataFrame) : Input data and predictions.
        - err_df (DataFrame) : False positive and negative rates for each year.
        - pan_str (str) : string indicating whether or not pan-susceptible isolates have
                            been removed.
        - prefix (str) : string defining prefix for the files in the saved plots.

    Returns
        - (altair chart) : visualization of the fraction of positive predictions in 
                            each year.
        - (altair chart) : visualization of the correlation between the RandomForest
                            and XGBoost predictions for each year.
    '''    

    ## Initialize a dataframe for plotting
    alt_df = pd.DataFrame({"Year":np.unique(df["Year"])})

    ## Define the classifier names
    classifier_names = df.columns[4:]

    ## Compute the fraction of positive predictions for each year
    for cname in classifier_names:
        year_fracs = getYearlyFractions(df[cname], df["Year"].to_numpy())
        alt_df[cname] = year_fracs

    ## Impute the colums of the yearly fraction dataframe to produce 
    # long-form data, then convert the fraction data into percentages.
    melt_alt_df = alt_df.melt("Year", var_name="Classifier", value_name="Fraction")
    melt_alt_df["Fraction"] = melt_alt_df["Fraction"] * 100

    ## Use the FPR and FNR values to approximate error margins
    melt_alt_df.set_index(["Year","Classifier"], inplace=True)
    for cname in classifier_names:
        fpr_vec = err_df[f"{cname} FPR"].to_numpy()
        fnr_vec = err_df[f"{cname} FNR"].to_numpy()

        idx_tup = (err_df.index, cname)
        melt_alt_df.loc[idx_tup,"min"] = melt_alt_df.loc[idx_tup,"Fraction"] * (1-fpr_vec)
        melt_alt_df.loc[idx_tup,"max"] = melt_alt_df.loc[idx_tup,"Fraction"] * (1+fnr_vec)
    melt_alt_df.reset_index(inplace=True)
    
    ## Visualize the predicted yearly fractions
    line = alt.Chart(melt_alt_df).mark_line().encode(
        x="Year:O",
        y=alt.Y("Fraction:Q", title="Proportion of Isolates (%)"),
        color="Classifier"
    )
    err = alt.Chart(melt_alt_df).mark_area(opacity=0.5).encode(
        x="Year:O",
        y=alt.Y("max:Q", title="Proportion of Isolates (%)"),
        y2=alt.Y2("min:Q", title="Proportion of Isolates (%)"),
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


    ## Create a correlation plot between the two predicted trends
    max_corner = max(alt_df["Random Forest"].max(), alt_df["XGBoost"].max())
    points = alt.Chart(alt_df).mark_point().encode(
        x=alt.X("Random Forest:Q").scale(domain=[0,max_corner]),
        y=alt.Y("XGBoost:Q").scale(domain=[0,max_corner])
    )
    corr_line = alt.Chart(alt_df).mark_rule(clip=True).encode(
        x = alt.datum(0),
        x2 = alt.datum(1),
        y = alt.datum(0),
        y2 = alt.datum(1)
    )
    corr_chart = alt.layer(
        points, 
        corr_line,
        data=alt_df
    )

    ## Save the charts
    corr_chart.save(f"output/{pan_str}/{prefix}.correlations.png")
    chart.save(f"output/{pan_str}/{prefix}.yearly_fractions.png")

    return chart, corr_chart




def trainClassifiers(df:pd.core.frame.DataFrame, pan_str:str, prefix:str, 
                            category_mapper:dict, atbs:list):
    '''
    Function to train and evaluate the Random Forest and XGBoost classifiers.

    Arguments
        - df (DataFrame) : DataFrame containing the raw data to use for training,
                            including antibiotics, years, and labels.
        - pan_str (str) : string determining whether or not pan-susceptible isolates 
                            have been removed or included in this dataset.
        - prefix (str) : file prefix for the outputs describing the model performance.
        - category_mapper (dict) : dictionary that maps the labels in the dataframe
                            to integers.
        - atbs (list) : names of the antibiotics to be used for training the models.

    Returns
        - (dict) : dictionary of classifiers trained on a subset of df.
    '''
    df = df.copy(deep=True)

    ## Split the data into a test and training set
    x = df[atbs].to_numpy()
    df["Label"] = df["Label"].map(category_mapper)
    y = df["Label"].to_numpy()
    year = df["Year"].to_numpy()
    names = df.index.to_numpy()
    (xs,ys,year_s,names_s), (xt,yt,year_t,names_t) = splitData(x,y,year,names, training_frac=0.75)

    ## Add which split the samples are in to the dataframe
    df.loc[names_s, "Split"] = "Training"
    df.loc[names_t, "Split"] = "Test"

    ## Load the whole-dataset models if they are already trained
    if os.path.exists(f"models/{pan_str}/{prefix}_random_forest.pkl") and os.path.exists(f"models/{pan_str}/{prefix}_xgboost.pkl"):
        with open(f"models/{pan_str}/{prefix}_random_forest.pkl","rb") as f:
            rf_model = pickle.load(f)
        with open(f"models/{pan_str}/{prefix}_xgboost.pkl","rb") as f:
            xgb_model = pickle.load(f)

        classifiers = {"Random Forest": rf_model,
                       "XGBoost": xgb_model}
        
    
    ## Initialize and train the models otherwise
    else:
        ## Initialize the classifiers
        classifiers = initClassifiers(verbosity=1)
        
        ## Fit each classifier to the training data
        for c_name,c in classifiers.items():
            print(f"\nFitting {c_name}:")
            c.fit(xs,ys)
   
        ## Save the models
        with open(f"models/{pan_str}/{prefix}_random_forest.pkl","wb") as f:
            pickle.dump(classifiers["Random Forest"].best_estimator_, f)
        with open(f"models/{pan_str}/{prefix}_xgboost.pkl","wb") as f:
            pickle.dump(classifiers["XGBoost"].best_estimator_, f)

    ## Initialize a dataframe for FPR and FNR
    err_df = pd.DataFrame({"Year":np.unique(year)})

    ## Compute the predictions, FPR, and FNR of each classifier
    for c_name,c in classifiers.items():
        ys_pred = c.predict(xs)
        yt_pred = c.predict(xt)
        df.loc[names_s,c_name] = ys_pred
        df.loc[names_t,c_name] = yt_pred

        _,_,_,_,_,conf,_ = testPerformance(yt, yt_pred, classifier_name=c_name, verbose=False)
        fpr = conf[0,1] / conf[0,:].sum()
        fnr = conf[1,0] / conf[1,:].sum()
        # err_df[c_name] = 
        err_df[f"{c_name} FPR"] = fpr
        err_df[f"{c_name} FNR"] = fnr

    ## Reset the index of the error dataframe
    err_df.set_index("Year", inplace=True)

    ## Return the models and the datsets
    return classifiers, df, err_df

