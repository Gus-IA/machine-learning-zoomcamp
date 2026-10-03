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


# linear regression

print(df_train.iloc[10])

xi = [453, 11, 86]

w0 = 7.17
w = [0.01, 0.04, 0.002]


def linear_regression(xi):
    n = len(xi)

    pred = w0

    for j in range(n):
        pred = pred + w[j] * xi[j]

    return pred


print(linear_regression(xi))

np.exp(12.312)


def dot(xi, w):
    n = len(xi)

    res = 0.0

    for j in range(n):
        res = res + xi[j] * w[j]

    return res


def linear_regression(xi):
    return w0 + dot(xi, w)


w_new = [w0] + w

print(w_new)


def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new)


print(linear_regression(xi))


xi = [453, 11, 86]

w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w

x1 = [1, 148, 24, 1385]
x2 = [1, 132, 25, 2031]
x10 = [1, 453, 1, 86]

X = [x1, x2, x10]
print(X)
X = np.array(X)
print(X)


def linear_regression(X):
    return X.dot(w_new)


print(linear_regression(X))


def train_linear_regression(X, y):
    pass


X = [
    [1, 148, 24, 1385],
    [1, 132, 25, 2031],
    [1, 453, 1, 86],
    [172, 25, 201],
    [142, 31, 86],
    [453, 31, 86],
    [158, 25, 185],
]

X = np.column_stack([x1, x2, x10])

ones = np.ones(X.shape[0])

X = np.column_stack([ones, X])

y = np.array([100, 200, 150, 250, 100, 200, 150, 250, 120])

XTX = X.T.dot(X)
XTX_inv = np.linalg.pinv(XTX)

print("X:", X.shape)
print("XTX:", XTX.shape)
print("XTX_inv:", XTX_inv.shape)

w_full = XTX_inv.dot(X.T).dot(y)

print(w_full)

w0 = w_full[0]
w = w_full[1:]

print(w0, w)


print(df_train.columns)

base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"]

X_train = df_train[base].values

X_train = df_train[base].fillna(0).isnull().sum()

w0, w = train_linear_regression(X_train)

w_pred = w0 + X_train.dot(w)
print(w_pred)

sns.histplot(w_pred, color="red", alpha=0.5, bins=50)
sns.histplot(X_train, color="blue", alpha=0.5, bins=50)


# RMSE
def rmse(y, y_pred):
    error = y - y_pred
    se = error**2
    mse = se.mean()
    return np.sqrt(mse)


# validating the model
base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"]

X_train = df_train[base].values

X_train = df_train[base].fillna(0).isnull().sum()

w0, w = train_linear_regression(X_train)

w_pred = w0 + X_train.dot(w)


def prepare_X(df):
    df_num = df[base]
    df_num.fillna(0).values
    X = df_num.values
    return X


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train)

X_train = prepare_X(df_val)
y_pred = w0 + X_train.dot(w)

rmse(X_train, y_pred)


# feature engineering
2017 - df_train.year


def prepare_X(df):
    df["age"] = 2017 - df_train.year
    features = base + ["age"]

    df_num = df[features]
    df_num.fillna(0).values
    X = df_num.values
    return X


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train)

X_train = prepare_X(df_val)
y_pred = w0 + X_train.dot(w)

rmse(X_train, y_pred)

sns.histplot(w_pred, color="red", alpha=0.5, bins=50)
sns.histplot(X_train, color="blue", alpha=0.5, bins=50)

# categorical variables

for v in [2, 3, 4]:
    df_train["num_doors_%" % v] = df_train["num_doors_4"] = (
        df_train.number_of_doors == v
    ).astype("int")


def prepare_X(df):
    df["age"] = 2017 - df_train.year
    features = base.append("age")

    for v in [2, 3, 4]:
        df_train["num_doors_%" % v] = df.number_of_doors["num_doors_4"] = (
            df_train.number_of_doors == v
        ).astype("int")
        features.append("num_doors_%s" % v)

    df_num = df[features]
    df_num.fillna(0).values
    X = df_num.values
    return X


prepare_X(df_train)

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train)

X_train = prepare_X(df_val)
y_pred = w0 + X_train.dot(w)

rmse(X_train, y_pred)

list(df.make.value_counts().head().index)


def prepare_X(df):
    df["age"] = 2017 - df_train.year
    features = base.append("age")

    for v in [2, 3, 4]:
        df_train["num_doors_%" % v] = df.make["num_doors_4"] = (
            df_train.number_of_doors == v
        ).astype("int")
        features.append("make_%s" % v)

    df_num = df[features]
    df_num.fillna(0).values
    X = df_num.values
    return X


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train)

X_train = prepare_X(df_val)
y_pred = w0 + X_train.dot(w)

rmse(X_train, y_pred)

# regularization

X = [[4, 4, 4], [3, 5, 5], [5, 1, 1], [5, 4, 4], [7, 5, 5], [4, 5, 5]]

X = np.array(X)
print(X)

y = [1, 2, 3, 1, 2, 3]


XTX = X.T.dot(X)
print(XTX)
XTX_inv = np.linalg.inv(XTX).dot(y)
print(XTX_inv)

XTX = [[1, 2, 2], [2, 1, 1.0000001], [2, 1.0000001, 1]]

XTX = np.array(XTX)
np.linalg.inv(XTX)


XTX = XTX + 0.01 * np.eye(3)
np.linalg.inv(XTX)


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


# tuning the model
for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(r, w0, score)
