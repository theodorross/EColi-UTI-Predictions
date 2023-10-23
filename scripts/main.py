import pandas as pd
import numpy as np
import os
import pickle

from dataproc import extractNORMdata, extractUTIdata
from classifierutils import splitData, initClassifiers, testClassifiers, predictClassifiers



if __name__=="__main__":

    '''
    Declare important parameters.
    '''
    antibiotics = ["Ceftazidim", "Ciprofloxacin", "Gentamicin"]


    '''
    Load the processed dataframes.
    '''
    ## Load the processed NORM dataframe
    if os.path.exists("data/processed-spreadsheets/NORM_data.csv"):
        norm_df = pd.read_csv("data/processed-spreadsheets/NORM_data.csv", index_col="Run accession")
    ## Process the raw NORM data if necessary
    else:
        norm_df = extractNORMdata("data/raw-spreadsheets/per_isolate_AST_DD_SIR_v4.xlsx", *antibiotics)
        norm_df.to_csv("data/processed-spreadsheets/NORM_data.csv")

    ## Load the processed UTI dataframe
    if os.path.exists("data/processed-spreadsheets/UTI_data.csv"):
        uti_df = pd.read_csv("data/processed-spreadsheets/UTI_data.csv", index_col="Unnamed: 0")
    ## Process the raw UTI data if necessary
    else:
        uti_df = extractUTIdata("data/raw-spreadsheets/20220324_E. coli NORM urin 2000-2021_no_metadata[2].xlsx", *antibiotics)
        uti_df.to_csv("./data/processed-spreadsheets/UTI_data.csv")

    
    ## Isolate years post methodology normalization by EUCAST
    norm_df = norm_df.loc[norm_df["Year"] >= 2011, :]
    # uti_df = uti_df.loc[uti_df["Year"] >= 2011, :]
    
    ## Force a consistent categorical encoding to the labels in the NORM dataframe
    labellist = norm_df["Label"].unique().tolist()
    labellist.sort()
    labellist = labellist[::-1]
    LABEL2CAT = {val:ix for ix,val in enumerate(labellist)}


    '''
    Train Classifier Models.
    '''
    ## Split the training data into training and test subsets
    x = norm_df[antibiotics].to_numpy()
    y = norm_df["Label"].map(LABEL2CAT).to_numpy()
    year = norm_df["Year"].to_numpy()
    (xs,ys,year_s), (xt,yt,year_t) = splitData(x,y,year, training_frac=0.8, seed=8415)

    ## Initialize the classifiers
    classifiers = initClassifiers(verbosity=1)
    
    ## Fit each classifier to the training data
    for c_name,c in classifiers.items():
        print(f"\nFitting {c_name}:")
        c.fit(xs,ys)

    ## Test the trained classifiers
    teststr, testchart, fpr_dict, fnr_dict = testClassifiers(classifiers, Test=(xt,yt,year_t), Training=(xs,ys,year_s))

    ## Use the classifiers to predict the ST-131 Clade C membership for the UTI data
    uti_dd = uti_df[antibiotics].to_numpy()
    uti_year = uti_df["Year"].to_numpy()
    predchart = predictClassifiers(classifiers, x=uti_dd, year=uti_year, fpr_dict=fpr_dict, fnr_dict=fnr_dict, uti_idx=uti_df.index)

    ## Save the models
    with open("models/random_forest.pkl","wb") as f:
        pickle.dump(classifiers["Random Forest"].best_estimator_, f)
    with open("models/xgboost.pkl","wb") as f:
        pickle.dump(classifiers["XGBoost"].best_estimator_, f)

    ## Save the visualizations
    testchart.save(f"output/yearly_fraction_predictions_training.png")
    testchart.save(f"output/yearly_fraction_predictions_training.html")
    predchart.save(f"output/yearly_fraction_predictions_unlabelled.png")
    predchart.save(f"output/yearly_fraction_predictions_unlabelled.html")

    ## Save the textual test performance output
    with open("output/classifier_metrics.txt","w") as f:
        f.write(teststr)


