import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

from dataproc import extractNORMdata, extractUTIdata, extractBSIdata
from classifierutils import testClassifiers, predictClassifiers, trainClassifiers, getYearlyFractions


if __name__=="__main__":

    '''
    Define antimicrobials to be used.
    '''
    antibiotics = ["Ceftazidim", "Ciprofloxacin", "Gentamicin"]

    '''
    Load the processed dataframes.
    '''
    print("loading data...")
    ## Load the processed NORM dataframe
    if os.path.exists(f"data/processed-spreadsheets/NORM_data.csv"):
        norm_df = pd.read_csv(f"data/processed-spreadsheets/NORM_data.csv", index_col="Run accession")
    ## Process the raw NORM data if necessary
    else:
        norm_df = extractNORMdata("data/raw-spreadsheets/per_isolate_AST_DD_SIR_v4.xlsx", 
                                    *antibiotics)
        norm_df.to_csv(f"data/processed-spreadsheets/NORM_data.csv")

    ## Load the processed UTI dataframe
    if os.path.exists(f"data/processed-spreadsheets/UTI_data.csv"):
        uti_df = pd.read_csv(f"data/processed-spreadsheets/UTI_data.csv", index_col="Unnamed: 0")
    ## Process the raw UTI data if necessary
    else:
        uti_df = extractUTIdata("data/raw-spreadsheets/20220324_E. coli NORM urin 2000-2021_no_metadata[2].xlsx", 
                                *antibiotics)
        uti_df.to_csv(f"./data/processed-spreadsheets/UTI_data.csv")
    
    ## Load the BSI dataframe
    if os.path.exists(f"data/processed-spreadsheets/BSI_data.csv"):
        bsi_df = pd.read_csv(f"data/processed-spreadsheets/BSI_data.csv", index_col="Unnamed: 0")
    ## Process the raw BSI data if necessary
    else:
        bsi_df = extractBSIdata("data/raw-spreadsheets/E_coli_2002_2021_BSI_exclude_WGS.xlsx",
                                *antibiotics)
        bsi_df.to_csv(f"./data/processed-spreadsheets/BSI_data.csv")
    
    ## Force a consistent categorical encoding to the labels in the NORM dataframe
    labellist = norm_df["Label"].unique().tolist()
    labellist.sort()
    labellist = labellist[::-1]
    LABEL2CAT = {val:ix for ix,val in enumerate(labellist)}

    ## Drop the singleton year/label combination in the NORM dataset
    drop_combo = ["131-C","2006"]
    drop_mask = (norm_df["Label"]=="131-C") & (norm_df["Year"]==2006)
    norm_df = norm_df[~drop_mask]

    ## Isolate years post methodology normalization by EUCAST
    norm_pre_2011 = norm_df.loc[norm_df["Year"] < 2011, :].copy()
    norm_post_2011 = norm_df.loc[norm_df["Year"] >= 2011, :].copy()

    uti_pre_2011 = uti_df.loc[uti_df["Year"] < 2011, :].copy()
    uti_post_2011 = uti_df.loc[uti_df["Year"] >= 2011, :].copy()

    bsi_pre_2011 = bsi_df.loc[bsi_df["Year"] < 2011, :].copy()
    bsi_post_2011 = bsi_df.loc[bsi_df["Year"] >= 2011, :].copy()

    print("UTI df:", uti_df.shape, uti_pre_2011.shape, uti_post_2011.shape)
    print("NORM df:", norm_df.shape, norm_pre_2011.shape, norm_post_2011.shape)
    print("BSI df:", bsi_df.shape, bsi_pre_2011.shape, bsi_post_2011.shape)


    '''
    Train Classifier Models.
    '''
    print("training models...")
    ## Train classifiers on the combined dataset
    kwargs = {"category_mapper":LABEL2CAT, "atbs":antibiotics}
    whole_classifiers, whole_norm_preds, whole_norm_err = trainClassifiers(norm_df, prefix="2006-2017", **kwargs)

    ## Train classifiers on the data before and after 2011 separately
    old_classifiers, old_norm_preds, old_norm_err = trainClassifiers(norm_pre_2011, prefix="2006-2010", **kwargs)
    new_classifiers, new_norm_preds, new_norm_err = trainClassifiers(norm_post_2011, prefix="2011-2017", **kwargs)
    
    ## Combine the split model predictions into one dataframe and force consistent ordering
    split_norm_preds = pd.concat([old_norm_preds, new_norm_preds])
    split_norm_preds = split_norm_preds.loc[whole_norm_preds.index]
    split_norm_err = pd.concat([old_norm_err, new_norm_err])

    ## Test the classifiers on the labelled data
    whole_tests = testClassifiers(whole_norm_preds, whole_norm_err, prefix="NORM-combined", write_files=True)
    split_tests = testClassifiers(split_norm_preds, split_norm_err, prefix="NORM-split", write_files=True)
    split_tests_old = testClassifiers(old_norm_preds, old_norm_err, prefix="NORM-split-old", write_files=True)
    split_tests_new = testClassifiers(new_norm_preds, new_norm_err, prefix="NORM-split-new", write_files=True)

    ## Save the predictions
    savecols = ["Year","Split","Label","XGBoost","Random Forest"]
    whole_norm_preds[savecols].to_csv(f"output/NORM-combined.predictions.csv")
    split_norm_preds[savecols].to_csv(f"output/NORM-split.predictions.csv")

    mask = (split_norm_preds["Year"] < 2011) & (split_norm_preds["Split"] == "Test")

    '''
    Utilize the classifiers on the unlabelled UTI data.
    '''
    print("predicting...")
    ## Isolate the data for prediction.
    uti_x = uti_df[antibiotics].to_numpy()
    old_uti_x = uti_pre_2011[antibiotics].to_numpy()
    new_uti_x = uti_post_2011[antibiotics].to_numpy()

    bsi_x = bsi_df[antibiotics].to_numpy()
    old_bsi_x = bsi_pre_2011[antibiotics].to_numpy()
    new_bsi_x = bsi_post_2011[antibiotics].to_numpy()

    ## Compute the predictions for each classifier
    for c_name in old_classifiers.keys():
        uti_df[c_name] = whole_classifiers[c_name].predict(uti_x)
        uti_pre_2011[c_name] = old_classifiers[c_name].predict(old_uti_x)
        uti_post_2011[c_name] = new_classifiers[c_name].predict(new_uti_x)

        bsi_df[c_name] = whole_classifiers[c_name].predict(bsi_x)
        bsi_pre_2011[c_name] = old_classifiers[c_name].predict(old_bsi_x)
        bsi_post_2011[c_name] = new_classifiers[c_name].predict(new_bsi_x)

    ## Combine the split model predictions into one dataframe and force consistent ordering
    split_uti_df = pd.concat([uti_pre_2011, uti_post_2011])
    split_uti_df = split_uti_df.loc[uti_df.index]

    split_bsi_df = pd.concat([bsi_pre_2011, bsi_post_2011])
    split_bsi_df = split_bsi_df.loc[bsi_df.index]

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
    bsi_true_fraction = getYearlyFractions(norm_df["Label"].map(LABEL2CAT), norm_df["Year"], "Sequenced BSI")
    
    ## Save the predictions
    savecols = ["Year","XGBoost","Random Forest"]
    uti_df[savecols].to_csv(f"output/UTI-combined.predictions.csv")
    split_uti_df[savecols].to_csv(f"output/UTI-split.predictions.csv")
    bsi_df[savecols].to_csv(f"output/BSI-combined.predictions.csv")
    split_bsi_df[savecols].to_csv(f"output/BSI-split.predictions.csv")

    ## Compute results of the chosen model
    keep_cols = split_uti_df.columns[:4].tolist() + ["Random Forest", "XGBoost"]
    predictClassifiers(split_uti_df[keep_cols], split_bsi_df[keep_cols], split_norm_err, prefix="UTI-BSI-predictions-split", truth_trend=bsi_true_fraction)
    predictClassifiers(uti_df[keep_cols], bsi_df[keep_cols], whole_norm_err, prefix="UTI-BSI-predictions-combined", truth_trend=bsi_true_fraction)    

    

