import pandas as pd
import numpy as np
import pickle

from classifierutils import testPerformance, getYearlyFractions


def val_converter(inp):
    try:
        return float(inp)
    except:
        return np.nan


if __name__ == "__main__":

    ## Load the UTi dataset
    uti_df = pd.read_excel("~/Desktop/Table S1 Edited Ørjan Rebecca FINAL.xlsx")
    uti_df.set_index("Sample accession", inplace=True)
    uti_df = uti_df[["ST", "Ceftazidim DD","Cipro         DD","Genta        DD"]]
    uti_df.rename(columns = {"Genta        DD":"Gentamicin",
                             "Cipro         DD":"Ciprofloxacin",
                             "Ceftazidim DD":"Ceftazidime"},
                   inplace=True)
    uti_df["Ceftazidime"] = uti_df["Ceftazidime"].apply(val_converter)
    uti_df["Ciprofloxacin"] = uti_df["Ciprofloxacin"].apply(val_converter)
    uti_df["Gentamicin"] = uti_df["Gentamicin"].apply(val_converter)
    uti_df.dropna(how="any", axis=0, inplace=True)
    
    ## Load the clade data
    clade_df = pd.read_excel('~/Desktop/FINAL Rev Table S1.xlsx',
                             sheet_name="Metadata",
                             usecols=["Sample accession","ST","ST131 clade","PP clade"],
                             index_col="Sample accession")
    
    # m_131 = clade_df.loc[uti_df.index, "ST"] == 131
    m1 = clade_df.loc[uti_df.index, "ST131 clade"] == "C1"
    m2 = clade_df.loc[uti_df.index, "ST131 clade"] == "C2"
    true_labs = (m1 | m2).astype(int)
    uti_df.loc[true_labs.index, "Label"] = true_labs
    
    clade_df["ST"] = clade_df["ST"].astype(str)
    clade_df["ST131 clade"] = clade_df["ST131 clade"].fillna('').astype(str)
    clade_df["Group"] = clade_df[["ST","ST131 clade"]].agg("-".join, axis=1)
    uti_df.loc[uti_df.index,"Group"] = clade_df.loc[uti_df.index,"Group"]

    ## Load the models to be used
    with open("models/2011-2017_xgboost.pkl","rb") as f:
        xgb_model = pickle.load(f)
    with open("models/2011-2017_random_forest.pkl","rb") as f:
        rf_model = pickle.load(f)
    with open("models/2011-2017_linear.pkl","rb") as f:
        lin_model = pickle.load(f)

    models = {"Linear":lin_model,
              "Random Forest":rf_model,
              "XGBoost":xgb_model}
    
    ## Make some predictions
    x = uti_df[["Ceftazidime","Ciprofloxacin","Gentamicin"]].to_numpy()
    y_true = uti_df["Label"].to_numpy()

    out_string = ""
    for m_name, model in models.items():
        print(f"\n{m_name}")
        y_scores = model.predict_proba(x)[:,1]
        performance_metrics = testPerformance(y_true, y_scores, classifier_name=m_name, verbose=False)
        y_pred = y_scores > 0.5
        print(uti_df.loc[y_pred,"Group"])
        out_string += performance_metrics[-1]

    ## Save the output to a text file
    with open("output/Genomic-UTI.classifier_metrics.txt","w") as f:
        f.write(out_string)
