# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#2 Importing the dataset
train_csv = pd.read_csv('train_100.csv')

#3 Training
x_train = train_csv.iloc[:,:-1]
y_train = train_csv.iloc[:, -1]
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(x_train,y_train)

print(grid.best_estimator_)

test_data = pd.read_csv("test_100.csv")
x = test_csv.iloc[:,:-1]
y_test = test_csv.iloc[:, -1]
grid_predictions = grid.predict(x_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
