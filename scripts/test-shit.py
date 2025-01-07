import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt

from sklearn.metrics import f1_score

from classifierutils import splitData


## Load and organize the data
df = pd.read_csv("data/processed-spreadsheets/NORM_data_include-pan.csv", index_col="Run accession")

labellist = df["Label"].unique().tolist()
labellist.sort()
labellist = labellist[::-1]
LABEL2CAT = {val:ix for ix,val in enumerate(labellist)}
df["Label"] = df["Label"].map(LABEL2CAT)
df_s, df_t = splitData(df, training_frac=0.75, stratify=df["Label"])

## Load the models
with open("models/include-pan/2006-2017_random_forest.pkl","rb") as f:
    old_rf = pickle.load(f)
with open("models//2006-2017_random_forest.pkl","rb") as f:
    new_rf = pickle.load(f)

## Isolate the data and compute inferences
x_s = df_s[["Ceftazidim","Ciprofloxacin","Gentamicin"]].to_numpy()
y_s = df_s["Label"].to_numpy()
x_t = df_t[["Ceftazidim","Ciprofloxacin","Gentamicin"]].to_numpy()
y_t = df_t["Label"].to_numpy()

old_pred_s = old_rf.predict(x_s)
new_pred_s = new_rf.predict(x_s)
old_pred_t = old_rf.predict(x_t)
new_pred_t = new_rf.predict(x_t)

print(f1_score(y_s, old_pred_s))
print(f1_score(y_s, new_pred_s))

print(f1_score(y_t, old_pred_t))
print(f1_score(y_t, new_pred_t))
