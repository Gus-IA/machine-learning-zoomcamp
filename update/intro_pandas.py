import numpy as np
import pandas as pd

# dataset
data = [
    ["Nissan", "Stanza", 1991, 138, 4, "MANUAL", "sedan", 2000],
    ["Hyundai", "Sonata", 2017, None, 4, "AUTOMATIC", "Sedan", 27150],
    ["Lotus", "Elise", 2010, 218, 4, "MANUAL", "convertible", 54990],
    ["GMC", "Acadia", 2017, 194, 4, "AUTOMATIC", "4dr SUV", 34450],
    ["Nissan", "Frontier", 2017, 261, 6, "MANUAL", "Pickup", 32340],
]

columns = [
    "Make",
    "Model",
    "Year",
    "Engine HP",
    "Engine Cylinders",
    "Transmission Type",
    "Vehicle_Style",
    "MSRP",
]

# creación del dataframe y visualizamos
df = pd.DataFrame(data, columns=columns)
print(df)


data = [
    {
        "Make": "Nissan",
        "Model": "Stanza",
        "Year": 1991,
        "Engine HP": 138.0,
        "Engine Cylinders": 4,
        "Transmission Type": "MANUAL",
        "Vehicle_Style": "sedan",
        "MSRP": 2000,
    },
    {
        "Make": "Hyundai",
        "Model": "Sonata",
        "Year": 2017,
        "Engine HP": None,
        "Engine Cylinders": 4,
        "Transmission Type": "AUTOMATIC",
        "Vehicle_Style": "Sedan",
        "MSRP": 27150,
    },
]

df = pd.DataFrame(data)
print(df)

df.head()

df.Make
df_engine = df["Engine HP"]
print(df_engine)

df_model = df[["Make", "Model", "MSRP"]]
print(df_model)

print(df.Make)

print(df.loc[1])

df_year = df["Year"] >= 2015
print(df_year)

df_price = df.MSRP.max()
print(df_price)

df_price_describe = df.MSRP.describe()
print(df_price_describe)

df_null = df.isnull().sum()
print(df_null)
