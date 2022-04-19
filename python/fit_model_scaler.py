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
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier

# Read data
df = pd.read_csv('../data/titanic_cleaned.csv', index_col=0)

# Split in training and testsets
x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1),
                                                    df['Survived'],
                                                    test_size=0.30,
                                                    random_state=42)

# Normalize the x data
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train[:] = scaler.fit_transform(x_train)
x_test[:] = scaler.fit_transform(x_test)



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

for i in range(1,71): # calculate error rates
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    error_rates.append(np.mean(pred != y_test))


plt.figure(figsize=(10,10))
plt.plot(range(1,71), error_rates, color="blue", marker='o', markerfacecolor='red')
plt.title('Error Rates vs. K-values (k-values 1 - 70)', size=20)
plt.xlabel('K-value', size=15)
plt.ylabel('Error rate', size=15)
#plt.savefig('../plots/knn_errorrates.png')
#plt.show()
plt.close()


# Make new model with k=63
knn_model_two = KNeighborsClassifier(n_neighbors=63)
knn_model_two.fit(x_train, y_train)
prediction_a = knn_model_two.predict(x_test)

# Evaluate model
print('\nEVALUATION OF SECOND MODEL (k=63):\n')
print('Classification report:\n', classification_report(y_test, prediction_a), '\n')
print('Confusion matrix:\n', confusion_matrix(y_test, prediction_a), '\n')
print('Allround score: ', round(knn_model_two.score(x_train, y_train) * 100, 2), '\n')


'''
Save the model and scaler to files
'''
filename_model = '../model_and_scaler/knn_titanic_model.sav'
joblib.dump(knn_model_two, filename_model)

filename_scaler = '../model_and_scaler/knn_titanic_scaler.sav'
joblib.dump(scaler, filename_scaler)





'''
Add new data and predict
'''
print(x_train.columns)
toms_data = {'Pclass':3, 'Age':30, 'SibSp':1, 'Parch':0, 'Fare':14, 'male':1, 'Q':1, 'S':0}
new_data = np.array([3, 30, 1, 0, 14, 1, 1, 0])
new_data = new_data.reshape(1, -1)
new_data = scaler.transform(new_data)

new_pred = knn_model_two.predict(new_data)

if new_pred[0] == 0:
    print('YOU DIED')
elif new_pred[0] == 1:
    print('You survived')
else:
    print('Something went wrong')


    










