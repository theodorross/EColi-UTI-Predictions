import pandas as pd
import numpy as np
import os

from dataproc import extractNORMdata, extractUTIdata
from classifierutils import splitData, initClassifiers



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
        norm_df = pd.read_csv("data/processed-spreadsheets/NORM_data.csv")
    ## Process the raw NORM data if necessary
    else:
        norm_df = extractNORMdata("data/raw-spreadsheets/per_isolate_AST_DD_SIR_v4.xlsx", *antibiotics)
        norm_df.to_csv("data/processed-spreadsheets/NORM_data.csv")

    ## Load the processed UTI dataframe
    if os.path.exists("data/processed-spreadsheets/UTI_data.csv"):
        uti_df = pd.read_csv("data/processed-spreadsheets/UTI_data.csv")
    ## Process the raw UTI data if necessary
    else:
        uti_df = extractUTIdata("data/raw-spreadsheets/20220324_E. coli NORM urin 2000-2021_no_metadata[2].xlsx", *antibiotics)
        uti_df.to_csv("./data/processed-spreadsheets/UTI_data.csv")

    
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
    xs,_,ys,xt,_,yt = splitData(x,y, training_frac=0.8)

    ## Initialize the classifiers
    classifiers = initClassifiers()
    
    ## Fit each classifier to the training data
    for c in classifiers.values():
        c.fit(xs,ys)


