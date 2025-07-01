# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# load data
import pandas as pd

data = pd.read_csv('data//train.csv')
df = data.copy()
df.sample(10)
# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df.info()
# %%
# check if there is any NaN in the dataset
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# convert categorical data into numerical data using one-hot encoding
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1.
df = pd.get_dummies(df)
df.sample(10)
# %% 
# separate the features and labels
X = df.drop(columns='Survived')
y = df['Survived']

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# %%
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}')
# %%
# build model
# build three classification models
# SVM, KNN, Random Forest
model_svm = SVC()
model_svm.fit(X_train, y_train)
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
# %%
# predict and evaluate
y_pred_svm = model_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {acc_svm:.4f}')
print(classification_report(y_test, y_pred_svm))
y_pred_knn = model_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {acc_knn:.4f}')
print(classification_report(y_test, y_pred_knn))
y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {acc_rf:.4f}')
print(classification_report(y_test, y_pred_rf))