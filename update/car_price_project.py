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
