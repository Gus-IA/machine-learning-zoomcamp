import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(url)

df.to_csv("data-week-3.csv", index=False)

print("Archivo descargado correctamente.")

df = pd.read_csv("data-week-3.csv")
print(df.head())

print(df.head().T)

df.columns = df.columns.str.lower().str.replace(" ", "_")

categorical_columns = list(df.dtypes[df.dtypes == "object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ", "_")

print(df.head().T)

tc = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)
print(df.churn.head())
(df.churn == "yes").astype(int).head()

print(tc)
