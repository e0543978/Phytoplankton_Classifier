# Importing Libraries

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# 2 Importing the dataset
train_csv = pd.read_csv('train_pca_100.csv', header=None)

# 3 Training
x_train = train_csv.iloc[:, :-1]
y_train = train_csv.iloc[:, -1]
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['poly', 'rbf', 'linear', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(x_train, y_train)

print(grid.best_estimator_)

test_csv = pd.read_csv("test_pca_100.csv", header=None)
x_test = test_csv.iloc[:, :-1]
y_test = test_csv.iloc[:, -1]
grid_predictions = grid.predict(x_test)
print(classification_report(y_test, grid_predictions))
