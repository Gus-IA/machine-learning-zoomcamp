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

from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

print(len(df_train), len(df_val), len(df_test))

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

# Exploratori and data analysis
df_full_train = df_full_train.reset_index(drop=True)
df_full_train.churn.value_counts(normalize=True)

# global_churn_rate = df_full_train.churn.mean()
# round(global_churn_rate, 2)

numerical = ["tenure", "monthlycharges", "totalcharges"]

categorical = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]

df_full_train[categorical].nunique()

# churn rate
df_full_train.head()

df_full_train["churn"] = (df_full_train["churn"] == "yes").astype(int)

global_churn = df_full_train.churn.mean()
print(global_churn)

churn_female = df_full_train[df_full_train.gender == "female"].churn.mean()
print(churn_female)

churn_male = df_full_train[df_full_train.gender == "male"].churn.mean()
print(churn_male)

df_full_train.partner.value_counts()

churn_partner = df_full_train[df_full_train.partner == "yes"].churn.mean()
print(churn_partner)

churn_no_partner = df_full_train[df_full_train.partner == "no"].churn.mean()
print(churn_no_partner)

print(global_churn - churn_partner)


# risk ratio
print(churn_no_partner / global_churn)

print(churn_partner / global_churn)

for c in categorical:
    print(c)
    df_group = df_full_train.groupby("gender").churn.agg(["mean", "count"])
    df_group["diff"] = df_group["mean"] - global_churn
    df_group["risk"] = df_group["mean"] / global_churn
    print(df_group)
    print()
    print()
