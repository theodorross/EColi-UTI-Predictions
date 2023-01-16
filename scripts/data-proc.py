import pandas as pd
import warnings
import numpy as np


def extractNORMdata(filepath, *antibiotics, remove_pan_susceptible=True):
    '''
    Extract and process AST data from the NORM data spreadsheets.  Isolates the disk-diffusion
    measurements of desired antibiotics and assigns a label to each isolate.  The labels 
    assigned are binary labels indicating whether or not the isolates belong to ST-131 clade C.
    
    Arguments:
        - filepath (str): relative path to the excel file containing the raw NORM data.
        - antibiotics (str): arbitrary number of antibiotics to include in the post-processed 
            dataset
        - remove_pan_susceptible (bool): flag to determine whether or not pan-susceptible 
            isolates are removed from the table.

    Returns:
        - pandas dataframe: post-processed dataframe containing labels and disk-diffusion
            measurements from the dataset.
    '''

    ## Load the file into a dataframe
    df = pd.read_excel(filepath)

    ## Set the run accession values as the index.
    df.set_index("Run accession", inplace=True)

    ## Strip whitespace from all columns.
    for col in df.columns:
        if isinstance(df[col][0], str):
            df[col] = df[col].str.strip()

    ## Filter out data that is pan-susceptible.
    if "susceptibility" in df.columns:
        if remove_pan_susceptible:
            keepidx = df["susceptibility"].isna().to_numpy().nonzero()[0]
            df = df.iloc[keepidx,:]
        df = df.drop(columns="susceptibility")
    
    ## Isolate ST131, Clade C isolates.
    # Define boolean masks for ST131 and clade C.
    st131_mask = df["ST"].to_numpy() == 131
    clade_c1_mask = df["Clade"].to_numpy() == "C1"
    clade_c2_mask = df["Clade"].to_numpy() == "C2"

    # Combine the masks and use them to generate a label array.
    clade_c_mask = np.logical_or(clade_c1_mask, clade_c2_mask)
    full_mask = np.logical_and(st131_mask, clade_c_mask)
    labels = ["131-C" if m else "other" for m in full_mask]

    df.insert(0, column="Label", value=labels)

    ## Isolate the zone diameter data and remove all other columns from the dataframe.
    columns = list(df.columns)
    all_atbs = []
    atb_start_idx = 8
    for s in columns[atb_start_idx:]:
        if "_" not in s:
            all_atbs.append(s)

    ## Isolate only zone diameter data of the antibiotics of interest.
    atbs_of_interest = []
    for atb in antibiotics:
        if atb not in all_atbs:
            warnings.warn(f"The specified antibiotic '{atb}' was not found in the datasheet.")
        else:
            atbs_of_interest.append(atb)
    df = df[["Label"] + atbs_of_interest]

    ## Drop rows with NaN values.
    df.dropna(inplace=True, axis=0)

    return df
 

def extractUTIdata(filepath, *antibiotics):
    '''
    
    '''
    
    ## Load the file into a dataframe.
    df = pd.read_excel(filepath)

    ## Isolate the zone diameter data and remove all other columns from the dataframe.
    columns = list(df.columns)
    all_atbs = []
    atb_start_idx = 8
    for s in columns[atb_start_idx:]:
        if "_" not in s:
            all_atbs.append(s)

    ## Isolate only zone diameter data of the antibiotics of interest.
    atbs_of_interest = []
    for atb in antibiotics:
        if atb not in all_atbs:
            warnings.warn(f"The specified antibiotic '{atb}' was not found in the datasheet.")
        else:
            atbs_of_interest.append(atb)
    df = df[atbs_of_interest]\
    
    ## Drop rows with NaN values.
    df.dropna(inplace=True, axis=0)
    
    return df


if __name__ == "__main__":


    ## Extract and preprocess datasets with the desired antibiotics.
    antibiotics = ["Ceftazidim", "Ciprofloxacin", "Gentamicin"]

    NORM_df = extractNORMdata("../data/raw-spreadsheets/per_isolate_AST_DD_SIR_v4.xlsx", *antibiotics)
    UTI_df = extractUTIdata("../data/raw-spreadsheets/20220324_E. coli NORM urin 2000-2021_no_metadata[2].xlsx", *antibiotics)

    ## Save the processed data to new files.
    NORM_df.to_csv("../data/processed-spreadsheets/NORM_data.csv")
    UTI_df.to_csv("../data/processed-spreadsheets/UTI_data.csv")
