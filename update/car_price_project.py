import pandas as pd

# descargamos el dataset
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"

df = pd.read_csv(url)
df.to_csv("data.csv", index=False)

print("Archivo descargado correctamente")

# lo cargamos y mostramos algunos datos
pd.read_csv("data.csv")
print(df.head())
df.columns.str.lower().str.replace(" ", "_")
df["Make"].str.lower().str.replace(" ", "_")

print(df.head())

# tipo de columna
strings = list(df.dtypes[df.dtypes == "object"])
print(strings)

# valores únicos por columna
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()

import matplotlib.pyplot as plt
import seaborn as sns

# distribución de precios menores a 100.000
sns.histplot(df.MSRP[df.MSRP < 100000], bins=50)
plt.show()

import numpy as np

# logaritmo
print(np.log([1, 10, 1000, 100000]))

# lo aplicamos
price_logs = np.log1p(df.MSRP)
print(price_logs)

# lo visualizamos
sns.histplot(price_logs, bins=50)
plt.show()

# missing values
print(df.isnull().sum())

n = len(df)

# separamos en validación, test y train
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

print(n_val, n_test, n_train)

df_val = df.iloc[n_train : n_train + n_val]
print(df_val)
df_test = df.iloc[n_val : n_val + n_test]
print(df_test)
df_train = df.iloc[n_train:]
print(df_train)

# mezclamos porque pueden estar ordenados..
idx = np.arange(n)

np.random.shuffle(idx)

print(idx)

# lo aplicamos y comprobamos
df_val = df.iloc[idx[n_train : n_train + n_val]]
print(df_val)
df_test = df.iloc[idx[n_val : n_val + n_test]]
print(df_test)
df_train = df.iloc[idx[n_train:]]
print(df_train)

print(len(df_train), len(df_val), len(df_test))

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
