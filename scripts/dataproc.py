import pandas as pd
import warnings
import numpy as np


def extractNORMdata(filepath, *antibiotics, remove_pan_susceptible=False):
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

    ## Create an ST-Clade column
    df["Group"] = df[["ST","Clade"]].apply(lambda row: "-".join(row.values.astype(str)), axis=1)
    df["Group"] = df["Group"].str.rstrip("-nan")
    
    ## Isolate ST131, Clade C isolates.
    # Define boolean masks for ST131 and clade C.
    # st131_mask = df["ST"].to_numpy() == 131                   ## CHANGE HERE TO RESTRICT TO ST131
    clade_c1_mask = df["Clade"].to_numpy() == "C1"
    clade_c2_mask = df["Clade"].to_numpy() == "C2"

    # Combine the masks and use them to generate a label array.
    clade_c_mask = np.logical_or(clade_c1_mask, clade_c2_mask)
    full_mask = clade_c_mask                                    ## CHANGE HERE TO RESTRICT TO ST131
    # full_mask = np.logical_and(st131_mask, clade_c_mask)      ## CHANGE HERE TO RESTRICT TO ST131
    labels = ["131-C" if m else "other" for m in full_mask]

    df.insert(0, column="Label", value=labels)

    ## Isolate the zone diameter data and remove all other columns from the dataframe.
    columns = list(df.columns)
    all_atbs = []
    atb_start_idx = 8
    for s in columns[atb_start_idx:]:
        if "_" not in s:
            all_atbs.append(s)

    ## Isolate only zone diameter data of the antibiotics of interest, label, and year.
    atbs_of_interest = []
    test_type_cols = []
    for atb in antibiotics:
        if atb not in all_atbs:
            warnings.warn(f"The specified antibiotic '{atb}' was not found in the datasheet.")
        else:
            atbs_of_interest.append(atb)
            test_type_cols.append(f"{atb}_ResType")

    ## Mask out the MIC tests
    restype_df = df[test_type_cols]
    MIC_mask = (restype_df=="MIC").any(axis="columns")
    df = df.loc[~MIC_mask,:]

    ## Select only the needed columns
    df = df[["Label","Year","Group"] + atbs_of_interest]

    ## Drop rows with NaN values.
    df.dropna(inplace=True, axis=0)

    return df
 

def extractUTIdata(filepath, *antibiotics, remove_pan_susceptible=False):
    '''
    Extract and preprocess the desired information from the UTI datasheet.  Isolates the disk-
    diffusion zone diameter measurements of the specified antibiotics.

    Arguments:
        - filepath (str): relative path to the excel file containing the raw NORM data.
        - antibiotics (str): arbitrary number of antibiotics to include in the post-processed 
            dataset

    Returns:
        - pandas dataframe: post-processed dataframe containingdisk-diffusion measurements 
            from the dataset.
    '''
    
    ## Load the file into a dataframe.
    df = pd.read_excel(filepath)

    ## Filter out pan-susceptible isolates
    if remove_pan_susceptible:
        # Isolate the columns of SIR data
        sir_cols = [col for col in df.columns if "SIR" in col]
        sir_df = df[sir_cols]

        # Compute the mask of pan-susceptible isolates
        mask_df = (sir_df == "S") | sir_df.isna()
        pan_mask = mask_df.all(axis="columns")

        # Mask out the pan-susceptible isolates
        df = df.loc[~pan_mask, :]


    ## Isolate the zone diameter data and remove all other columns from the dataframe.
    columns = list(df.columns)
    all_atbs = []
    atb_start_idx = 8
    for s in columns[atb_start_idx:]:
        if "_" not in s:
            all_atbs.append(s)

    ## Isolate only zone diameter data of the antibiotics of interest and year.
    atbs_of_interest = []
    for atb in antibiotics:
        if atb not in all_atbs:
            warnings.warn(f"The specified antibiotic '{atb}' was not found in the datasheet.")
        else:
            atbs_of_interest.append(atb)

    df = df[["Prove_aar"] + atbs_of_interest]
    df.rename(columns={"Prove_aar":"Year"}, inplace=True)

    ## Reset the indices to match with the row numbers of the raw data table
    df.set_index(df.index+2, inplace=True)

    ## Drop rows with NaN values.
    df.dropna(inplace=True, axis=0)
    
    return df


def extractBSIdata(filepath, *antibiotics):
    '''
    Extract and preprocess the desired information from the BSI datasheet.  Isolates the disk-
    diffusion zone diameter measurements of the specified antibiotics.

    Arguments:
        - filepath (str): relative path to the excel file containing the raw NORM data.
        - antibiotics (str): arbitrary number of antibiotics to include in the post-processed 
            dataset

    Returns:
        - pandas dataframe: post-processed dataframe containingdisk-diffusion measurements 
            from the dataset.
    '''
    raw_df = pd.read_excel(filepath)
    
    ## Remove rows with sequenced data and ones with MIC instead of disk diffusion
    keep_mask = raw_df["Sequenced"].isna()
    for atb in antibiotics:
        _m = raw_df[f"{atb}_ResType"] == "Sonediameter"
        keep_mask = keep_mask & _m
    df = raw_df.loc[keep_mask]

    ## Keep relevant columns
    df = df[["Prove_aar"] + list(antibiotics)]

    ## Rename the year column
    df.rename(mapper={"Prove_aar":"Year"}, inplace=True, axis='columns')
    return df


if __name__ == "__main__":


    ## Extract and preprocess datasets with the desired antibiotics.
    antibiotics = ["Ceftazidim", "Ciprofloxacin", "Gentamicin"]

    NORM_df = norm_df = extractNORMdata("data/raw-spreadsheets/per_isolate_AST_DD_SIR_v4.xlsx", *antibiotics)
    # UTI_df = extractUTIdata("./data/raw-spreadsheets/20220324_E. coli NORM urin 2000-2021_no_metadata[2].xlsx", *antibiotics)
    # BSI_df = extractBSIdata("data/raw-spreadsheets/E_coli_2002_2021_BSI_exclude_WGS.xlsx", *antibiotics)
    

    ## Save the processed data to new files.
    NORM_df.to_csv("data/processed-spreadsheets/NORM_data.csv")
    # UTI_df.to_csv("data/processed-spreadsheets/UTI_data.csv")
    # BSI_df.to_csv("data/processed-spreadsheets/BSI_data.csv")
