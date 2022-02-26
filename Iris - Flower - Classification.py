#   IRIS FLOWER CLASSIFICATION

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Accessing the Dataset csv file using Pandas
dataset = pd.read_csv('dataset iris.data')
print(dataset.head())

print(dataset.info())

print(dataset.describe())

print(dataset[" Class"].value_counts())

# Editing the dataset inorder to remove the unnecessary columns
le = LabelEncoder()
dataset[' Class'] = le.fit_transform(dataset[' Class'])
print("                           Dataset Head")
print(dataset.head())
print("                           Dataset Middle")
print(dataset[50:55])
print("                           Dataset Last")
print(dataset.tail())

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X = dataset.drop(columns=[' Class'])
Y = dataset[' Class']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# Using Logistic Regression Algorithm to classify the flowers
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 3000)

# Model Training
model.fit(x_train, y_train)

'''LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)'''

# Accuracy of the Trained Model
print("Accuracy: ",model.score(x_test, y_test) * 100)

#Prediction of the species from the input vector
X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
prediction = model.predict(X_new)
print("Prediction of Species: {}".format(prediction))

for val in prediction:
    if val == 0:
        print("Iris-setosa")
    elif val == 1:
        print("Iris-versicolor")
    else:
        print("Iris-virginica ")
