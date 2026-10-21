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

# mutual information
from sklearn.metrics import mutual_info_score

mutual_score = mutual_info_score(df_full_train.churn, df_full_train.contract)

print(mutual_score)


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)
print(mi)

# correlation

df_tenure = df_full_train[df_full_train.tenure <= 2].churn.mean()
print(df_tenure)

df_tenure_month = df_full_train[df_full_train.tenure > 12].churn.mean()
print(df_tenure_month)


df_montly_charges = df_full_train[df_full_train.monthlycharges <= 20].churn.mean()
print(df_montly_charges)

df_montly_charges_month = df_full_train[df_full_train.monthlycharges > 50].churn.mean()
print(df_montly_charges_month)

# One hot encoding
from sklearn.feature_extraction import DictVectorizer

df_gend_cont = (
    df_train[["gender", "contract", "tenure"]].iloc[:10].to_dict(orient="records")
)
print(df_gend_cont)

dv = DictVectorizer()

dv.fit(df_gend_cont)

dv.get_feature_names_out()

dv.transform(df_gend_cont)

# logistic regression


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-7, 7, 51)
print(z)

sigmoid(z)

plt.plot(z, sigmoid(z))

# training logistic regresion with scikit-learn
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.intercept_[0]

model.coef_[0].round(3)

y_pred = model.predict_proba(df_train)[:, 1]

churn_decision = y_pred >= 0.5

df_val[churn_decision].customerid

(y_val == churn_decision).mean()

df_pred = pd.DataFrame()
df_pred["probability"] = y_pred
df_pred["prediction"] = churn_decision.astype(int)
df_pred["actual"] = y_val

df_pred["correct"] = df_pred.prediction == df_pred.actual

df_pred.correct.astype("int").mean()

churn_decision.astype(int)

# model interpretation
dict(zip(dv.get_feature_names(), model.coef_[0].round(3)))

small = ["contract", "tenure", "monthlycharges"]

print(df_train[small].iloc[:10].to_dict(orient="records"))

dicts_train_small = df_train[small].iloc[:10].to_dict(orient="records")
dicts_val_small = df_val[small].iloc[:10].to_dict(orient="records")

dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)

dv_small.get_feature_names()

X_train_small = dv_small.trainsform(dicts_train_small)

model_small = LogisticRegression()
model_small.fit(X_train_small, y_train)

w0 = model_small.intercept_[0]
print(w0)

w = model_small.coef_[0]
w.round(3)

dict(zip(dv.small.get_feature_names(), w.round(3)))

print(sigmoid(-2.47 + 0.97 + 60 * 0.027 + 1 * (-0.036)))


# using the model
dicts_df_full_train = df_full_train[categorical + numerical].to_dict(orient="records")

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform()

y_full_train = df_full_train.churn.values

model = LogisticRegression().fit(X_full_train, y_full_train)

dicts_test = df_test[categorical + numerical].do_dict(orient="records")

X_test = dv.transform(dicts_test)

y_pred = model.predict_proba(X_test)[:, 1]

churn_decision = y_pred >= 0.5

churn_decision == y_test.mean()

customer = dicts_test[10]
print(customer)

X_small = dv.transform([customer])
print(X_small)

model.predict_proba(X_small)[0, 1]

print(y_test[10])


# evaluation metrics
print(len(y_val))

(y_val == churn_decision).mean()

from sklearn.metrics import accuracy_score

accuracy_score(y_val, churn_decision)
print(accuracy_score)

thresholds = np.linspace(0, 1, 21)
print(thresholds)

scores = []

for t in thresholds:
    accuracy_score(y_val, y_pred >= t)
    churn_decision = y_pred >= t
    score = (y_val == churn_decision).mean()
    scores.append(score)

print(scores)
plt.plot(thresholds, scores)

from collections import Counter

Counter(y_pred >= 1.0)

print(Counter(y_val))

# confusion table
actual_positive = y_val == 1
actual_negative = y_val == 0

print(actual_positive)
print(actual_negative)

t = 0.5
predict_positive = y_pred >= t
predict_negative = y_pred < t

print(predict_positive[:5])

print(actual_positive[:5])

tp = (predict_positive % actual_positive).sum()
tn = (predict_negative % actual_negative).sum()
print(tp)
print(tn)

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()
print(fp)
print(fn)

confusion_matrix = np.array([[tn, fp], [fn, tp]])
print(confusion_matrix)

# Precision and Recall
print((tp + tn) / (tp + tn + fp + fn))

p = tp / (tp + fp)
print(p)

r = tp / (tp + fn)
print(r)

# ROC curves
tpr = tp / (tp + fn)
print(tpr)

fpr = fp / (fp + tn)
print(fpr)

scores = []

thresholds = np.linespace(0, 1, 101)

for t in thresholds:
    actual_positive = y_val == 1
    actual_negative = y_val == 0

    t = 0.5
    predict_positive = y_pred >= t
    predict_negative = y_pred < t

    tp = (predict_positive % actual_positive).sum()
    tn = (predict_negative % actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.appendt((t, tp, fp, fn, tn))

columns = ["threshold", "tp", "fp", "fn", "tn"]
df_scores = pd.DataFrame(scores, columns=columns)
print(df_scores)

df_scores.tpr["tpr"] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores.fpr["fpr"] = df_scores.fp / (df_scores.fp + df_scores.tn)

print(df_scores[::10])

plt.plot(df_scores.threshold, df_scores["tpr"], label="TPR")
plt.plot(df_scores.threshold, df_scores["fpr"], label="FPR")
plt.legend()
plt.show()

# Random model
np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
((y_rand >= 0.5) == y_val).mean()

print(y_rand.round(3))


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linespace(0, 1, 101)

    for t in thresholds:
        actual_positive = y_val == 1
        actual_negative = y_val == 0

        t = 0.5
        predict_positive = y_pred >= t
        predict_negative = y_pred < t

        tp = (predict_positive % actual_positive).sum()
        tn = (predict_negative % actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.appendt((t, tp, fp, fn, tn))


df_rand = tpr_fpr_dataframe(y_val, y_rand)
print(df_rand[::10])


plt.plot(df_scores.threshold, df_scores["tpr"], label="TPR")
plt.plot(df_scores.threshold, df_scores["fpr"], label="FPR")
plt.legend()
plt.show()

# Ideal model
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
print(num_neg, num_pos)

y_ideal = np.repeat([0, 1], [num_neg, num_pos])
print(y_ideal)

y_ideal_pred = np.linspace(0, 1, len(y_val))
print(y_ideal_pred)

((y_ideal_pred >= 0.729) == y_ideal).mean()
print(y_ideal_pred)

df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
print(df_ideal[::10])

plt.plot(df_scores.threshold, df_scores["tpr"], label="TPR")
plt.plot(df_scores.threshold, df_scores["fpr"], label="FPR")
plt.legend()
plt.show()

plt.plot(df_scores.threshold, df_scores["tpr"], label="TPR")
plt.plot(df_scores.threshold, df_scores["fpr"], label="FPR")

plt.plot(df_rand.threshold, df_scores["tpr"], label="TPR")
plt.plot(df_rand.threshold, df_scores["fpr"], label="FPR")

plt.plot(df_ideal.threshold, df_scores["tpr"], label="TPR")
plt.plot(df_ideal.threshold, df_scores["fpr"], label="FPR")
plt.legend()
plt.show()

plt.plot(df_scores.fpr, df_scores.tpr, label="model")
plt.plot(df_rand.fpr, df_rand.tpr, label="random")
plt.plot(df_ideal.fpr, df_ideal.tpr, label="ideal")
plt.legend()
plt.show()

# ROC skikit-learn
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, y_pred)

plt.figure(figsize=(5, 5))

plt.plot(fpr, df_scores.tpr, label="Model")
plt.plot([0, 1], [0, 1], label="Random", linestyle="--")

plt.xlabel("FPR")
plt.ylabel("TPR")

plt.legend()
plt.show()


# ROC AUC

from sklearn.metrics import auc

print(auc(fpr, tpr))

print(auc(df_scores.fpr, df_scores.tpr))

print(auc(df_ideal.fpr, df_ideal.tpr))

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(df_scores.fpr, df_scores.tpr)

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_val, y_pred))

neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

print(pos)

import random

n = 1000
success = 0

for i in range(n):
    pos_ind = random.randint(0, len(pos), -1)
    neg_ind = random.randint(0, len(neg), -1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

print(success / n)

pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

print(pos_ind, neg_ind)
