import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

import pandas as pd

url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv"

df = pd.read_csv(url)

df.to_csv("CreditScoring.csv", index=False)

print("Archivo descargado correctamente.")
print(df.head())

df.columns = df.columns.str.lower()
print(df.columns)
print(df.status.value_counts())

df.status.map({1: "ok", 2: "default", 0: "unk"})
print(df.head())

home_values = {
    1: "rent",
    2: "owner",
    3: "private",
    4: "ignore",
    5: "parents",
    6: "other",
    0: "unk",
}

df.home = df.home.map(home_values)

marital_values = {
    1: "single",
    2: "married",
    3: "widow",
    4: "separated",
    5: "divorced",
    0: "unk",
}

df.marital = df.marital.map(marital_values)

records_values = {1: "no", 2: "yes", 0: "unk"}

df.records = df.records.map(records_values)

job_values = {1: "fixed", 2: "partime", 3: "freelance", 4: "others", 0: "unk"}

df.job = df.job.map(job_values)

print(df.head())
print(df.describe().round())

for c in ["income", "assets", "debt"]:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

print(df.describe().round())

print(df.status.value_counts())

print(df[df.status != "unk"].reset_index(drop=True))

from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.status == "default").astype("int").values
y_val = (df_val.status == "default").astype("int").values
y_test = (df_test.status == "default").astype("int").values

del df_train["status"]
del df_val["status"]
del df_test["status"]

print(df_train)


# decision trees
def assess_risk(client):
    if client["records"] == "yes":
        if client["job"] == "parttime":
            return "default"
        else:
            if client["assets"] > 6000:
                return "ok"
            else:
                return "default"


xi = df_train.iloc[0].to_dict()

assess_risk(xi)

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

train_dicts = df_train.to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

print(dv.get_feature_names_out())

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

val_dicts = df_val.fillna(0).to_dict(orient="records")
X_val = dv.transform(val_dicts)

y_pred = dt.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)

y_pred = dt.predict_proba(X_train)[:, 1]
roc_auc_score(y_val, y_pred)

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print("train:", auc)

y_pred = dt.predict_proba(X_train)[:, 1]
roc_auc_score(y_val, y_pred)
print("val:", auc)

from sklearn.tree import export_text

print(export_text(dt, feature_names=dv.get_feature_names_out()))

# decision tree learning algorith
data = [
    [8000, "default"],
    [2000, "default"],
    [0, "default"],
    [5000, "ok"],
    [5000, "ok"],
    [4000, "ok"],
    [9000, "ok"],
    [3000, "default"],
]

df_example = pd.DataFrame(data, columns=["assets", "status"])
print(df_example)

print(df_example.sort_values("assets"))

Ts = [2000, 3000, 4000, 5000, 8000]

from Ipython.display import display

for T in Ts:
    df_left = df_example[df_example.assets <= T]
    df_right = df_example[df_example.assets > T]

    display(df_left)
    print(df_left.status.value_counts(normalize=True))
    display(df_right)
    print(df_left.status.value_counts(normalize=True))

    print()

data = [
    [8000, 3000, "default"],
    [2000, 1000, "default"],
    [0, 1000, "default"],
    [5000, 1000, "ok"],
    [5000, 1000, "ok"],
    [4000, 1000, "ok"],
    [9000, 500, "ok"],
    [3000, 2000, "default"],
]

thresholds = {"assets": [0, 2000, 3000, 4000, 5000, 8000], "debt": [500, 1000, 2000]}

for feature, Ts in thresholds.items():
    print("###########################")
    print(feature)
    for T in Ts:
        print(T)
        df_left = df_example[df_example[feature] <= T]
        df_right = df_example[df_example[feature] > T]

        display(df_left)
        print(df_left.status.value_counts(normalize=True))
        display(df_right)
        print(df_right.status.value_counts(normalize=True))

        print()
    print("###########################")

# decision tree parameter tuning

for d in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    print("%4s -> %.3f" % (d, auc))

scores = []

for d in [4, 5, 6]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, s, auc))

columns = ["max_depth", "min_samples_leaf", "auc"]
df_scores = pd.DataFrame(scores, columns=columns)
scores.head()

df_scores.sort_values(by="auc", ascending=False).head()

df_scores_pivot = df_scores.pivot(index="min_samples_leaf", columns=["max_depth"])

sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")

# ensembles and random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)
rf.fit_(X_train, y_train)


rf.predict_proba(X_val)[:, 1]

y_pred = rf.predict_proba(X_val[:, 1])
roc_auc_score(y_val, y_pred)

scores = []
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit_(X_train, y_train)

    y_pred = rf.predict_proba(X_val[:, 1])
    auc = roc_auc_score(y_val, y_pred)
    scores.append((n, auc))

df_scores = pd.DataFrames(scores, columns=["n_estimators", "auc"])
plt.plot(df_scores.n_estimators, df_scores.auc)


scores = []
for n in range(5, 10, 15):
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
        rf.fit_(X_train, y_train)

        y_pred = rf.predict_proba(X_val[:, 1])
        auc = roc_auc_score(y_val, y_pred)
        scores.append((d, n, auc))

columns = ["max_depth", "n_estimators", "auc"]
df_scores = pd.DataFrames(scores, columns)
df.scores.head()

for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plt(df_subset.n_estimators, df_subset.auc, label="max_depth=%d" % d)

plt.legend()
