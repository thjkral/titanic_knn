#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:33:01 2022

@author: tom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  matplotlib.ticker import FuncFormatter

# load the data
df = pd.read_csv('../data/train.csv', index_col=0)

'''
Clean the raw data
'''
# visualization of missing data
plt.figure(figsize=(10,10))
sns.heatmap(df.isna(), yticklabels=False, cmap='viridis', cbar=False)
plt.title('Missing values in the Titanic passenger data', size=20)
plt.xlabel('Columns', size=15)
plt.ylabel('Rows', size=15)
#plt.savefig('../plots/rawData.png')
#plt.show()
plt.close()

# drop the 'Cabin' column
df.drop('Cabin', inplace=True, axis=1)

# fill missing 'Age' values with the mean age of their respective class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)

# remove rows with missing Embarked data
df.dropna(subset=['Embarked'], inplace=True)

# revisualize data
plt.figure(figsize=(10,10))
sns.heatmap(df.isna(), yticklabels=False, cmap='viridis', cbar=False)
plt.title('Titanic passenger data after cleaning the data', size=20)
plt.xlabel('Columns', size=15)
plt.ylabel('Rows', size=15)
#plt.savefig('../plots/cleanedData.png')
#plt.show()
plt.close()


'''Exploratory Data Analysis'''

# survivalrate for genders
plt.figure(figsize=(10,10))
genderCount = sns.countplot(x='Survived', data=df, hue='Sex', palette='coolwarm')
plt.title('Number of casualties and survivors per gender', size=20)
plt.xlabel('')
plt.ylabel('Count', size=15)
genderCount.set_xticklabels(['Perished','Survived'], size=15)
plt.legend(title='Gender',fontsize=13, title_fontsize=15)
#plt.savefig('../plots/survivalGender')
#plt.show()
plt.close()


# rate of perished people per ages
plt.figure(figsize=(20,5))
ageCount = sns.countplot(x='Age', data=df[df['Survived']==0])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
plt.title('Casualties per passenger age', size=20)
plt.xlabel('Age', size=15)
plt.ylabel('Count', size=15)
#plt.savefig('../plots/survivalAge.png')
#plt.show()
plt.close()


# show correlations between columns
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Correlation between passenger attributes', size=20)
plt.xticks(size=15)
plt.yticks(size=15)
#plt.savefig('../plots/corr.png')
#plt.show()
plt.close()


# average ticketprices per passenger class
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Ticket prices per passenger class', size=20)
plt.xlabel('Class', size=15)
plt.ylabel('Price paid', size=15)
#plt.savefig('../plots/faresPerClass.png')
#plt.show()
plt.close()

print(df.groupby('Pclass').mean()['Fare'])

'''Convert to final product and export'''

# convert categorical data
sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)

df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df = pd.concat([df,sex,embark],axis=1)



# save cleaned data as CSV
df.to_csv('../data/titanic_cleaned.csv')






