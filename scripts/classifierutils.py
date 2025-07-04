import numpy as np
import pandas as pd
import altair as alt
import os
import pickle

import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from scipy.stats import pearsonr, PermutationMethod

from matplotlib import pyplot as plt


def splitData(df, training_frac=0.8, seed=8431, stratify=None):
    '''
    Split data into training and test datasets.

    Arguments
        - x (numpy array) : data samples, with the first index corresponding to individual samples.
        - y (numpy array) : data labels.
        - training_frac (float in [0,1]) : fraction of the data that will be kept in the training set.
        - seed (float or int) : random seed used for the index shuffling operation, kept constant to maintain 
                                constant datsets between tests.
    Returns
        - (tuple) : training splits of each input array
        - (tuple) : test splits of each input array
    '''

    df_train, df_test = sk.model_selection.train_test_split(df, train_size=training_frac, 
                                                            random_state=seed, stratify=stratify)
    
    return df_train, df_test



def initClassifiers(verbosity=0, rf_params=None, xgb_params=None, lin_params=None):
    '''
    Initializes two pre-defined classifiers chosen for performance comparisson.

    Inputs
        - verbosity (int) : controls the verbosity of sklearn GridSearchCV objects
        - rf_params (dict) : dictionary of hyperparameters for the Random Forest classifier
        - xgb_params (dict) : dictionary of hyperparameters for the XGBoost classifier

    Returns
        - (dict) : Dictionary with classifier names for keys corresponding to classifier objects.
        - (dict) : Dictionary with the same keys but boolean values corresponding to whether or not
                    the data should be normalized for the classifier corresponding to the key.
    '''

    ## Return objects to perform hyperparameter grid searches
    if (rf_params is None) or (xgb_params is None) or (lin_params is None):
        ## Initialize the Random Forest classifier for cross-validation
        rf = RandomForestClassifier(class_weight="balanced")      
        rf_grid = {"n_estimators":[10,50,100,250,500,1000],
                   "class_weight":["balanced",None]}  
        rf_cv = GridSearchCV(estimator=rf, param_grid=rf_grid, scoring="f1", 
                             cv=5, verbose=verbosity, refit=True)
        
        ## Initialize the XGBoost classifier for cross-validation
        xgb = XGBClassifier(eval_metric="mlogloss")         
        xgb_grid = {"n_estimators":[5,10,50,100,250],
                    "learning_rate":[0.01,0.1,0.5,1.0,1.5],
                    "max_depth":[1,2,4,6]}
        xgb_cv = GridSearchCV(estimator=xgb, param_grid=xgb_grid, scoring="f1", 
                              cv=5, verbose=verbosity, refit=True)

        ## Initialize a baseline linear classifier
        # lin = LogisticRegression(solver="newton-cholesky")
        lin = LogisticRegression(solver="saga")
        lin_grid = {"penalty":["l2","l1","elasticnet"],
                    "class_weight":["balanced",None]}
        lin_cv = GridSearchCV(estimator=lin, param_grid=lin_grid, scoring="f1",
                              cv=5, verbose=verbosity, refit=True)
        
        ## Create a classifier dictionary
        classifier_dict = {"Random Forest": rf_cv, "XGBoost": xgb_cv, "Linear": lin_cv}

    ## Return the classifiers alone instead of grid search objects
    else:
        rf = RandomForestClassifier(**rf_params)
        xgb = XGBClassifier(eval_metric="mlogloss", **xgb_params)
        lin = LogisticRegression(solver="saga", **lin_params)
        classifier_dict = {"Random Forest": rf, "XGBoost": xgb, "Linear": lin}

    return classifier_dict



def testPerformance(y_true:np.ndarray, y_scores:np.ndarray, classifier_name:str=None, verbose:bool=True, labels=None, thresh=0.5):
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
    ## Compute categorical predictions
    y_pred = (y_scores > thresh).astype(int)

    ## Compute performance metrics
    acc = np.mean(y_true==y_pred)
    f1 = sk.metrics.f1_score(y_true, y_pred, labels=labels)
    conf = sk.metrics.confusion_matrix(y_true, y_pred, labels=labels)
    precision = sk.metrics.precision_score(y_true, y_pred, average="binary", zero_division=np.nan)
    recall = sk.metrics.recall_score(y_true, y_pred, average="binary", zero_division=np.nan)
    tn,fp,fn,tp = conf.ravel()
    sensitivity = recall
    specificity = tn / (tn + fp)

    roc_auc = sk.metrics.roc_auc_score(y_true, y_scores)
    avg_prec = sk.metrics.average_precision_score(y_true, y_scores)

    ## Print performance metrics
    out_str = ""
    if classifier_name is not None: out_str += f"\n\n\t{classifier_name} Classifier\n"
    out_str += f"Accuracy: \t{acc:.5f}\n"
    out_str += f"F1-Score: \t{f1:.5f}\n"
    out_str += f"Precision: \t{precision:.5f}\n"
    out_str += f"Recall:  \t{recall:.5f}\n"
    out_str += f"Specificity: \t{specificity:.5f}\n"
    out_str += f"Sensitivity: \t{sensitivity:.5f}\n"
    out_str += f"ROC AUC: \t{roc_auc:.5f}\n"
    out_str += f"Avg Precision:\t{avg_prec}\n"
    out_str += f"Confusion Matrix: \n{conf}\n"

    if verbose:
        print(out_str, end="")

    return acc, f1, precision, recall, specificity, sensitivity, roc_auc, avg_prec, conf, out_str



def getYearlyFractions(label:np.ndarray, year:np.ndarray, colname:str) -> pd.core.frame.DataFrame:
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
    year_masks = {yr:year==yr for yr in year_axis}
    year_frac = pd.DataFrame(index=year_axis, columns=[colname])
    for yr,mask in year_masks.items():
        year_frac.loc[yr,colname] = np.sum(label[mask]==1)/mask.sum()
    return year_frac



def testClassifiers(df:pd.core.frame.DataFrame, err_df:pd.core.frame.DataFrame, prefix:str, write_files:bool):
    '''
    Function to test the performance of input classifiers.  Computes various metrics 
    including accuracy, F1-score, precision, recall, sensitivity, and specificity.  
    The false-positive and -negative rates are also computed for computing error bands
    in the yearly fraction plots.  These yearly fraction plots are also computed 
    independently for the test and training sets.

    Arguments
        - df (DataFrame) : pandas DataFrame containing the predictions on the Test and
                            Training datasets. Must contain columns ["Split","Label",
                            "Year"] as well as any model names used for prediction.
        - classifier_names (list) : list of classifier names to be tested. Must correspond
                            with the names of the model prediction columns in df.
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
    df = df.copy(deep=True)

    ## Initialize performance summary data structures.
    charts = []
    printstr = ""

    ## Isolate the classifier names
    classifier_names = ["Linear","Random Forest","XGBoost"]

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
        y = split_df["Label"].to_numpy()
        year = split_df["Year"].to_numpy()

        ## Add a header to the summary text.
        printstr += "\n\n" + "="*56 + "\n"
        printstr += "="*21 + f" {split:12s} " +"="*21 + "\n"
        printstr += "="*56

        # Compute the true fraction of samples belonging to 131-C
        alt_df = getYearlyFractions(y, year, "Truth")
        year_axis = alt_df.index.to_numpy()

        for cname in classifier_names:

            ## Extract the model predictions
            y_scores = split_df[cname].to_numpy()
            y_pred = (y_scores > 0.5).astype(int)

            ## Compute the performance metrics for the current dataset-classifier combination
            performance_metrics = testPerformance(y, y_scores, cname, verbose=False)
            printstr += performance_metrics[-1]

            ## Compute predicted fractions of 131-C each year
            pred_year_frac = getYearlyFractions(y_pred, year, cname)
            alt_df = pd.merge(alt_df, pred_year_frac, how='outer', left_index=True, right_index=True)

            ## Print the false postive STs
            false_pos_mask = (y_pred==1) & (y==0)
            false_pos_list = split_df.loc[false_pos_mask,"Group"].value_counts()
            false_pos_singletons = false_pos_list[false_pos_list == 1].index.tolist()
            printstr += "\nFalse Positives:\n"
            printstr += str(false_pos_list[false_pos_list>1])
            printstr += f"\nSingletons: {false_pos_singletons}\n"
            # print(f"\n{split} {cname} false postives")
            # print(false_pos_list[false_pos_list>1])
            # print("Singletons:", false_pos_singletons)

            ## Add predicted information to the output string
            printstr += "\n\tYear  |  Predicted Fraction | True Fraction\n\t" + "-"*43 + "\n"
            for jx,yr in enumerate(year_axis):
                printstr += f"\t{yr}  |\t{alt_df.loc[yr,cname]:3f} \t| \t {alt_df.loc[yr,'Truth']:3f}\n"

            ## Compute the Pearson correlation coeffiecient between the truth and 
            #  predicted fraction values
            r_val = pearsonr(alt_df[cname].astype(float), alt_df["Truth"].astype(float), 
                            alternative="greater", method=PermutationMethod())
            printstr += f"\nPearson R: {r_val.statistic:5f}\nR P-value: {r_val.pvalue}\n"

            ## Loop through fold prediction columns
            fold_metric_df = pd.DataFrame(index=range(5), columns=["Accuracy", "F1-Score", "Precision", 
                                                                   "Recall", "Specificity", "Sensitivity",
                                                                   "ROC AUC", "Avg Precision"])
            for fold in range(5):
                fold_name = f"{cname} fold{fold}"
                fold_preds = split_df[fold_name].to_numpy()

                fold_metrics = testPerformance(y, fold_preds, fold_name, verbose=False)
                fold_metric_df.loc[fold,:] = fold_metrics[:8]

            printstr += f"\n\t{cname} cross-folds\n"
            fold_means = fold_metric_df.mean(axis=0)
            fold_stds = fold_metric_df.std(axis=0)
            for metric in fold_means.index:
                printstr += f"{metric:12s}:\t{fold_means.loc[metric]:5f} +/- {fold_stds.loc[metric]:5f}\n"


        ## Add a column to specify the data group
        alt_df["Dataset"] = split

        ## Save correlation plot data
        corr_df = pd.concat([corr_df, alt_df[corr_cols]])

        ## Impute the colums of the yearly fraction dataframe to produce 
        #  long-form data, then convert fractions into percentage.
        alt_df.index.name = "Year"
        alt_df.reset_index(inplace=True)
        alt_df = alt_df[trend_cols].melt("Year", var_name="Classifier", value_name="Fraction")
        alt_df["Fraction"] = alt_df["Fraction"] * 100
        
        ## Use the FDR and FOR values to approximate error margins
        alt_df.set_index(["Year","Classifier"], inplace=True)
        # for cname in classifier_names:
        for cname in classifier_names:
            fdr_vec = err_df[f"{cname} FDR"].to_numpy()
            for_vec = err_df[f"{cname} FOR"].to_numpy()

            idx_tup = (err_df.index, cname)
            alt_df.loc[idx_tup,"min"] = alt_df.loc[idx_tup,"Fraction"] * (1-fdr_vec)
            alt_df.loc[idx_tup,"max"] = alt_df.loc[idx_tup,"Fraction"] * (1+for_vec)

        # Undo the indexing
        alt_df.reset_index(inplace=True)

        ## Visualize the fraction predictions
        c_domain = classifier_names+["Truth"]
        c_range = ["#1f77b4","#ff7f0e","#2ca02c","#000000"]
        line = alt.Chart(alt_df).mark_line().encode(
            x="Year:O",
            y=alt.Y("Fraction:Q", title="Proportion of Isolates (%)"),
            color=alt.Color("Classifier:N", sort=c_domain).scale(domain=c_domain, range=c_range)
        ).properties(
            title=split
        )
        err_band = alt.Chart(alt_df).mark_errorband().encode(
            x="Year:O",
            y=alt.Y("max:Q", title="Proportion of Isolates (%)"),
            y2=alt.Y2("min:Q", title="Proportion of Isolates (%)"),
            color=alt.Color("Classifier:N", sort=c_domain, legend=alt.Legend(symbolType="stroke", symbolOpacity=1)).scale(domain=c_domain, range=c_range)
        )
        chrt = err_band+line
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
    # .configure_axis(
    #     labelFontSize=18,
    #     titleFontSize=18,
    #     labelLimit=400
    # ).configure_legend(
    #     labelFontSize=18,
    #     titleFontSize=20,
    #     titleLimit=400,
    #     labelLimit=600
    # ).configure_header(
    #     labelFontSize=20
    # ).configure_title(
    #     fontSize=22
    # )

    ## Write the outputs to files if specified
    if write_files:
        # Write the text output
        with open(f"output/{prefix}.classifier_metrics.txt","w") as f:
            f.write(printstr)

        # Save the yearly trend chart
        output_chart.save(f"output/{prefix}.yearly_fractions.png", ppi=300)

        # Save the yearly trend correlation chart
        corr_chart.save(f"output/{prefix}.correlations.png", ppi=300)
        
    return printstr, output_chart, corr_chart



def predictClassifiers(uti_df:pd.core.frame.DataFrame, bsi_df:pd.core.frame.DataFrame, 
                       err_df:pd.core.frame.DataFrame, prefix:str, truth_trend:pd.core.frame.DataFrame):
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

    ## Define the best classifier
    # cname = "Random Forest"
    cname = "XGBoost"

    ## Use the truth trend as an initial dataframe
    alt_df = truth_trend.copy()

    ## Define the classifier names
    classifier_names = uti_df.columns[4:]

    ## Compute the fraction of positive predictions for each year
    # for cname in classifier_names:
    uti_year_fracs = getYearlyFractions(uti_df[cname], uti_df["Year"].to_numpy(), colname="Predicted UTI")
    bsi_year_fracs = getYearlyFractions(bsi_df[cname], bsi_df["Year"].to_numpy(), colname="Predicted BSI")
    alt_df = pd.merge(alt_df, uti_year_fracs, left_index=True, right_index=True, how='outer')
    alt_df = pd.merge(alt_df, bsi_year_fracs, left_index=True, right_index=True, how='outer')

    ## Impute the colums of the yearly fraction dataframe to produce 
    # long-form data, then convert the fraction data into percentages.
    alt_df.index.name = "Year"
    alt_df.reset_index(inplace=True)
    melt_alt_df = alt_df.melt("Year", var_name="Classifier", value_name="Fraction")
    melt_alt_df["Fraction"] = melt_alt_df["Fraction"] * 100

    ## Use the FDR and FOR values to approximate error margins
    melt_alt_df.set_index(["Year","Classifier"], inplace=True)
    fdr_vec = err_df[f"{cname} FDR"].to_numpy()
    for_vec = err_df[f"{cname} FOR"].to_numpy()
    
    melt_alt_df.insert(loc=1, column="min", value=None)
    melt_alt_df.insert(loc=2, column="max", value=None)

    for _m in ["Predicted UTI", "Predicted BSI"]:
        idx_tup = (err_df.index, _m)
        melt_alt_df.loc[idx_tup,"min"] = melt_alt_df.loc[idx_tup,"Fraction"] * (1-fdr_vec)
        melt_alt_df.loc[idx_tup,"max"] = melt_alt_df.loc[idx_tup,"Fraction"] * (1+for_vec)
    melt_alt_df.reset_index(inplace=True)
    
    ## Visualize the predicted yearly fractions
    c_domain = melt_alt_df["Classifier"].unique().tolist()
    c_range = ["#000000","#1f77b4","#ff7f0e"]
    line = alt.Chart(melt_alt_df).mark_line().encode(
        x="Year:O",
        y=alt.Y("Fraction:Q", title="Proportion of Isolates (%)"),
        color=alt.Color("Classifier:N", title="Trend").scale(domain=c_domain, range=c_range)
    )
    err = alt.Chart(melt_alt_df).mark_errorband().encode(
        x="Year:O",
        y=alt.Y("max:Q", title="Proportion of Isolates (%)"),
        y2=alt.Y2("min:Q", title="Proportion of Isolates (%)"),
        color=alt.Color("Classifier:N", title="Trend", legend=alt.Legend(symbolType="stroke", symbolOpacity=1))
    )
    chart = err+line
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
    max_corner = max(alt_df["Predicted UTI"].max(), alt_df["Predicted BSI"].max())
    cor_df = pd.melt(alt_df, id_vars=["Year", "Sequenced BSI"], value_vars=["Predicted UTI", "Predicted BSI"], 
                     var_name="Dataset", value_name="Fraction")
    cor_df["Dataset"] = cor_df["Dataset"].str.replace("Predicted ", "")
    points = alt.Chart(cor_df).mark_point().encode(
        x=alt.X("Sequenced BSI:Q").scale(domain=[0,max_corner]),
        y=alt.Y("Fraction:Q", title="Predicted Fraction").scale(domain=[0,max_corner]),
        color = alt.Color("Dataset:N")
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
    corr_chart.save(f"output/{prefix}.correlations.png", ppi=300)
    chart.save(f"output/{prefix}.yearly_fractions.png", ppi=300)

    return chart, corr_chart




def trainClassifiers(data_df:pd.core.frame.DataFrame, prefix:str, 
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
        - (dict) : dictionary of classifiers trained on a df.
    '''
    df = data_df.copy(deep=True)

    ## Split the data into a test and training set
    df["Label"] = df["Label"].map(category_mapper)
    stratify = [f"y{yr}_l{lab}" for (yr,lab) in zip(df["Year"], df["Label"])]
    df_s, df_t = splitData(df, training_frac=0.75, stratify=stratify)

    xs = df_s[atbs].to_numpy()
    ys = df_s["Label"].to_numpy()
    year_s = df_s["Year"].to_numpy()
    names_s = df_s.index.to_numpy()

    xt = df_t[atbs].to_numpy()
    yt = df_t["Label"].to_numpy()
    year_t = df_t["Year"].to_numpy()
    names_t = df_t.index.to_numpy()

    ## Add which split the samples are in to the dataframe
    df.loc[names_s, "Split"] = "Training"
    df.loc[names_t, "Split"] = "Test"

    ## Load the whole-dataset models if they are already trained
    classifiers = {}
    if os.path.exists(f"models/{prefix}_linear.pkl"):
        with open(f"models/{prefix}_linear.pkl","rb") as f:
            lin_model = pickle.load(f)
        classifiers["Linear"] = lin_model

    if os.path.exists(f"models/{prefix}_random_forest.pkl"):
        with open(f"models/{prefix}_random_forest.pkl","rb") as f:
            rf_model = pickle.load(f)
        classifiers["Random Forest"] = rf_model
    
    if os.path.exists(f"models/{prefix}_xgboost.pkl"):
        with open(f"models/{prefix}_xgboost.pkl","rb") as f:
            xgb_model = pickle.load(f)
        classifiers["XGBoost"] = xgb_model
        
    
    ## Initialize and train the models otherwise
    ## Initialize the classifiers
    fresh_classifiers = initClassifiers(verbosity=1)
    
    ## Fit each classifier to the training data if one isn't already saved
    for c_name,c in fresh_classifiers.items():
        if c_name not in classifiers.keys():
            print(f"\nFitting {c_name}:")
            c.fit(xs,ys)
            classifiers[c_name] = c

            # Save the model 
            model_filename = c_name.lower().replace(" ","_")
            with open(f"models/{prefix}_{model_filename}.pkl","wb") as f:
                pickle.dump(c, f)
   
        ## Save the models
        # with open(f"models/{prefix}_linear.pkl","wb") as f:
        #     pickle.dump(classifiers["Linear"], f)
        # with open(f"models/{prefix}_random_forest.pkl","wb") as f:
        #     pickle.dump(classifiers["Random Forest"], f)
        # with open(f"models/{prefix}_xgboost.pkl","wb") as f:
        #     pickle.dump(classifiers["XGBoost"], f)

    ## Print selected hyperparameters
    print(f"\nLinear model hyperparameters: {prefix}")
    print(classifiers["Linear"].best_params_)

    print(f"Random Forest hyperparameters: {prefix}")
    print(classifiers["Random Forest"].best_params_)

    print(f"XGBoost hyperparameters: {prefix}")
    print(classifiers["XGBoost"].best_params_)

    ## Initialize a dataframe for FDR and FOR
    # err_df = pd.DataFrame({"Year":np.unique(year)})
    err_df = pd.DataFrame({"Year":df["Year"].unique()})

    ## Initialize feature importance frame
    importance_df = pd.DataFrame(columns=atbs, index=["Final Model","CV Mean","CV Std"]+[f"Fold {_ix}" for _ix in range(5)])

    ## Compute the predictions, FDR, and FOR of each classifier
    for c_name,c in classifiers.items():
        ys_pred = c.predict_proba(xs)[:,1]
        yt_pred = c.predict_proba(xt)[:,1]
        df.loc[names_s,c_name] = ys_pred
        df.loc[names_t,c_name] = yt_pred

        # Compute confusion matrix
        _,_,_,_,_,_,_,_,conf,_ = testPerformance(yt, yt_pred, classifier_name=c_name, verbose=False)
        tn, fp, fn, tp = conf.ravel()
        FDR = fp / (tp + fp)
        FOR = fn / (tn + fn)
        err_df[f"{c_name} FDR"] = FDR
        err_df[f"{c_name} FOR"] = FOR

        # Print the Random Forest feature importances
        if c_name == "Random Forest":
            importance_df.loc["Final Model",atbs] = c.best_estimator_.feature_importances_
            # print(f"RF Importance:", c.best_estimator_.feature_importances_)

    ## Train and save the models on each cross-fold
    folds = StratifiedKFold(n_splits=5).split(xs, ys)
    lin_hyperparams = classifiers["Linear"].best_params_
    rf_hyperparams = classifiers["Random Forest"].best_params_
    xgb_hyperparams = classifiers["XGBoost"].best_params_
    lin_fold_models = {}
    rf_fold_models = {}
    xgb_fold_models = {}

    for ix,(ks,kt) in enumerate(folds):
        # Isolate the training data in this fold
        fold_x = xs[ks]
        fold_y = ys[ks]

        # Initialize and train the models
        _models = initClassifiers(rf_params=rf_hyperparams, 
                                  xgb_params=xgb_hyperparams, 
                                  lin_params=lin_hyperparams)
        _models["Linear"].fit(fold_x, fold_y)
        _models["Random Forest"].fit(fold_x, fold_y)
        _models["XGBoost"].fit(fold_x, fold_y)

        # Store the trained models
        lin_fold_models[ix] = _models["Linear"]
        rf_fold_models[ix] = _models["Random Forest"]
        xgb_fold_models[ix] = _models["XGBoost"]

    ## Compute predictions for k-fold trained sub-models
    for c_name, c_dict in zip(["Linear","Random Forest","XGBoost"], [lin_fold_models,rf_fold_models,xgb_fold_models]):
        for kx,c in c_dict.items():
            _ys_pred = c.predict_proba(xs)[:,1]
            _yt_pred = c.predict_proba(xt)[:,1]
            df.loc[names_s, f"{c_name} fold{kx}"] = _ys_pred
            df.loc[names_t, f"{c_name} fold{kx}"] = _yt_pred

            # Print Random Forest importances
            if c_name == 'Random Forest':
                importance_df.loc[f"Fold {kx}",atbs] = c.feature_importances_
                # print(f"fold{kx}:", c.feature_importances_)
    
    ## Aggregate importance values for the cross-folds
    importance_df.loc["CV Mean",atbs] = importance_df.iloc[3:,:].mean(axis=0)
    importance_df.loc["CV Std",atbs] = importance_df.iloc[3:,:].std(axis=0)
    print(importance_df)

    ## Reset the index of the error dataframe
    err_df.set_index("Year", inplace=True)

    ## Return the models and the datsets
    return classifiers, df, err_df

