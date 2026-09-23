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
