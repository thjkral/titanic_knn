#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:07:11 2022

@author: tom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Read data
df = pd.read_csv('../data/titanic_cleaned.csv', index_col=0)

# Scale the data
def min_max_scaling(column):
    return (column - column.min()) / (column.max() - column.min())

df_norm = df
df_norm['Age'] = min_max_scaling(df_norm['Age'])
df_norm['SibSp'] = min_max_scaling(df_norm['SibSp'])
df_norm['Parch'] = min_max_scaling(df_norm['Parch'])
df_norm['Fare'] = min_max_scaling(df_norm['Fare'])


# Split in training and testsets
x_train, x_test, y_train, y_test = train_test_split(df_norm.drop('Survived', axis=1),
                                                    df_norm['Survived'],
                                                    test_size=0.30,
                                                    random_state=42)

'''
K-Nearest Neighbors
'''

# Make a first model
knn_model_one = KNeighborsClassifier(n_neighbors=3)
knn_model_one.fit(x_train, y_train)
prediction = knn_model_one.predict(x_test)

# Evaluate model
print('\nEVALUATION OF FIRST MODEL (k=3):\n')
print('Classification report:\n', classification_report(y_test, prediction), '\n')
print('Confusion matrix:\n', confusion_matrix(y_test, prediction), '\n')
print('Allround score: ', round(knn_model_one.score(x_train, y_train) * 100, 2), '\n')


# Testing multiple k-values
error_rates = []

for i in range(1,41): # calculate error rates
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    error_rates.append(np.mean(pred != y_test))


plt.figure(figsize=(10,10))
plt.plot(range(1,41), error_rates, color="blue", marker='o', markerfacecolor='red')
plt.title('Error Rates vs. K Value', size=20)
plt.xlabel('K-value', size=15)
plt.ylabel('Error rate', size=15)
#plt.savefig('../plots/knn_errorrates.png')
#plt.show()
plt.close()


# Make new model with k=36
knn_model_two = KNeighborsClassifier(n_neighbors=36)
knn_model_two.fit(x_train, y_train)
prediction_a = knn_model_two.predict(x_test)

# Evaluate model
print('\nEVALUATION OF SECOND MODEL (k=36):\n')
print('Classification report:\n', classification_report(y_test, prediction_a), '\n')
print('Confusion matrix:\n', confusion_matrix(y_test, prediction_a), '\n')
print('Allround score: ', round(knn_model_two.score(x_train, y_train) * 100, 2), '\n')

'''
Add new data and predict
'''

#toms_data = {'Pclass':3, 'Age':30, 'SibSp':1, 'Parch':0, 'Fare':14, 'male':1, 'Q':1, 'S':0}
tom_array = np.array([3,30,1,0,14,1,1,0])
tom_array = tom_array.reshape(1, -1)

toms_rediction = knn_model_two.predict(tom_array)










