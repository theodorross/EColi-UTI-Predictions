import pandas as pd
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

from dataproc import extractNORMdata, extractUTIdata
from classifierutils import splitData, initClassifiers, testClassifiers, predictClassifiers, trainClassifiers, getYearlyFractions



if __name__=="__main__":

    '''
    Declare important parameters.
    '''
    antibiotics = ["Ceftazidim", "Ciprofloxacin", "Gentamicin"]
    remove_pan = True
    if remove_pan:
        pan_str = "remove-pan"
    else:
        pan_str = "include-pan"


    '''
    Load the processed dataframes.
    '''
    print("loading data...")
    ## Load the processed NORM dataframe
    if os.path.exists(f"data/processed-spreadsheets/NORM_data_{pan_str}.csv"):
        norm_df = pd.read_csv(f"data/processed-spreadsheets/NORM_data_{pan_str}.csv", index_col="Run accession")
    ## Process the raw NORM data if necessary
    else:
        norm_df = extractNORMdata("data/raw-spreadsheets/per_isolate_AST_DD_SIR_v4.xlsx", 
                                    *antibiotics, remove_pan_susceptible=remove_pan)
        norm_df.to_csv(f"data/processed-spreadsheets/NORM_data_{pan_str}.csv")

    ## Load the processed UTI dataframe
    if os.path.exists(f"data/processed-spreadsheets/UTI_data_{pan_str}.csv"):
        uti_df = pd.read_csv(f"data/processed-spreadsheets/UTI_data_{pan_str}.csv", index_col="Unnamed: 0")
    ## Process the raw UTI data if necessary
    else:
        uti_df = extractUTIdata("data/raw-spreadsheets/20220324_E. coli NORM urin 2000-2021_no_metadata[2].xlsx", 
                                *antibiotics, remove_pan_susceptible=remove_pan)
        uti_df.to_csv(f"./data/processed-spreadsheets/UTI_data_{pan_str}.csv")
    
    ## Force a consistent categorical encoding to the labels in the NORM dataframe
    labellist = norm_df["Label"].unique().tolist()
    labellist.sort()
    labellist = labellist[::-1]
    LABEL2CAT = {val:ix for ix,val in enumerate(labellist)}

    ## Isolate years post methodology normalization by EUCAST
    norm_pre_2011 = norm_df.loc[norm_df["Year"] < 2011, :].copy()
    norm_post_2011 = norm_df.loc[norm_df["Year"] >= 2011, :].copy()

    uti_pre_2011 = uti_df.loc[uti_df["Year"] < 2011, :].copy()
    uti_post_2011 = uti_df.loc[uti_df["Year"] >= 2011, :].copy()


    '''
    Train Classifier Models.
    '''
    print("training models...")
    ## Train classifiers on the combined dataset
    kwargs = {"pan_str":pan_str, "category_mapper":LABEL2CAT, "atbs":antibiotics}
    whole_classifiers, whole_norm_preds, whole_norm_err = trainClassifiers(norm_df, prefix="2006-2017", **kwargs)
    
    ## Train classifiers on the data before and after 2011 separately
    old_classifiers, old_norm_preds, old_norm_err = trainClassifiers(norm_pre_2011, prefix="2006-2010", **kwargs)
    new_classifiers, new_norm_preds, new_norm_err = trainClassifiers(norm_post_2011, prefix="2011-2017", **kwargs)
    
    ## Combine the split model predictions into one dataframe and force consistent ordering
    split_norm_preds = pd.concat([old_norm_preds, new_norm_preds])
    split_norm_preds = split_norm_preds.loc[whole_norm_preds.index]
    split_norm_err = pd.concat([old_norm_err, new_norm_err])

    ## Test the classifiers on the labelled data
    whole_tests = testClassifiers(whole_norm_preds, whole_norm_err, pan_str=pan_str, prefix="NORM-combined", write_files=True)
    split_tests = testClassifiers(split_norm_preds, split_norm_err, pan_str=pan_str, prefix="NORM-split", write_files=True)
    split_tests_old = testClassifiers(old_norm_preds, old_norm_err, pan_str=pan_str, prefix="NORM-split-old", write_files=True)
    split_tests_new = testClassifiers(new_norm_preds, new_norm_err, pan_str=pan_str, prefix="NORM-split-new", write_files=True)

    ## Save the predictions
    savecols = ["Year","Label","XGBoost","Random Forest"]
    whole_norm_preds[savecols].to_csv(f"output/{pan_str}/NORM-combined.predictions.csv")
    split_norm_preds[savecols].to_csv(f"output/{pan_str}/NORM-split.predictions.csv")


    '''
    Utilize the classifiers on the unlabelled UTI data.
    '''
    print("predicting...")
    ## Isolate the data for prediction.
    uti_x = uti_df[antibiotics].to_numpy()
    old_uti_x = uti_pre_2011[antibiotics].to_numpy()
    new_uti_x = uti_post_2011[antibiotics].to_numpy()

    ## Compute the predictions for each classifier
    for c_name in old_classifiers.keys():
        uti_df[c_name] = whole_classifiers[c_name].predict(uti_x)
        uti_pre_2011[c_name] = old_classifiers[c_name].predict(old_uti_x)
        uti_post_2011[c_name] = new_classifiers[c_name].predict(new_uti_x)

    ## Combine the split model predictions into one dataframe and force consistent ordering
    split_uti_df = pd.concat([uti_pre_2011, uti_post_2011])
    split_uti_df = split_uti_df.loc[uti_df.index]

    ## Add the needed years to the FPR and FNR dataframes
    for y in uti_df["Year"].unique():
        if y not in split_norm_err.index.tolist():
            if y >= 2011:
                split_norm_err.loc[y] = split_norm_err.loc[2011]
            else:
                split_norm_err.loc[y] = split_norm_err.loc[2010]

        if y not in whole_norm_err.index.tolist():
            whole_norm_err.loc[y] = whole_norm_err.loc[2011]

    ## Compute the true yearly fractions from the BSI data
    bsi_true_fraction = getYearlyFractions(norm_df["Label"].map(LABEL2CAT), norm_df["Year"], "BSI Ground Truth")
    
    ## Save the predictions
    savecols = ["Year","XGBoost","Random Forest"]
    uti_df[savecols].to_csv(f"output/{pan_str}/UTI-combined.predictions.csv")
    split_uti_df[savecols].to_csv(f"output/{pan_str}/UTI-split.predictions.csv")

    ## Compute results of the chosen model
    best_model = "Random Forest"
    keep_cols = split_uti_df.columns[:4].tolist() + [best_model]
    predictClassifiers(split_uti_df[keep_cols], split_norm_err, pan_str=pan_str, prefix="UTI-split", truth_trend=bsi_true_fraction)
    predictClassifiers(uti_df[keep_cols], whole_norm_err, pan_str=pan_str, prefix="UTI-combined", truth_trend=bsi_true_fraction)

    

