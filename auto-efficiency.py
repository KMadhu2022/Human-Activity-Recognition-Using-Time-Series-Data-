import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tree import base
import metrics

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Removing rows in horsepower where val = ?
data = data[data['horsepower'] != '?'].reset_index(drop = True)

# Removing car name because it is unique to each car and will lead to high variance
data = data.drop('car name', axis = 1)

# Setting target attr
y = data['mpg']
X = data.drop('mpg', axis = 1)

X.rename(columns={'cylinders': 0}, inplace = True)
X.rename(columns={'displacement': 1}, inplace = True)
X.rename(columns={'horsepower': 2}, inplace = True)
X.rename(columns={'weight': 3}, inplace = True)
X.rename(columns={'acceleration': 4}, inplace = True)
X.rename(columns={'model year': 5}, inplace = True)
X.rename(columns={'origin': 6}, inplace = True)

# Train-test split and converting all X (input) and y (output) to real
X_train = pd.DataFrame(X[:275].reset_index(drop = True), dtype = np.float64)
y_train = pd.Series(y[:275].reset_index(drop = True), dtype = np.float64, name = None)
X_test = pd.DataFrame(X[275:].reset_index(drop = True), dtype = np.float64)
y_test = pd.Series(y[275:].reset_index(drop = True), dtype = np.float64, name = None)

# Hyperparameters
k = 8 # no of folds for cross validation
index = X_train.shape[0]//k
depths = [1,2,3,4,5,6,7,8,9,10]
optimum_depth = 0
min_error = np.inf

# Finding the optimum depth of decision tree using cross validation because we want the best performance of both trees on the dataset
for depth in depths:
  total_error = 0
  for i in range(k):
    X_train2 = pd.concat((X_train[0 : i*index], X_train[(i+1)*index : ]), axis = 0).reset_index(drop = True)
    X_validation = X_train[index*i : (i+1)*index].reset_index(drop = True)
    y_train2 = pd.concat((y_train[0 : i*index], y_train[(i+1)*index : ]), axis = 0).reset_index(drop = True)
    y_validation = y_train[index*i : (i+1)*index].reset_index(drop = True)

    model = DecisionTreeRegressor(max_depth = depth)
    model.fit(X_train2, y_train2)
    y_pred = model.predict(X_validation)
    total_error += metrics.rmse(y_pred, y_validation)

  if total_error/k < min_error: # if error for this depth is min, set opt_depth = this depth
    optimum_depth = depth
    min_error = total_error/k

# Using our decision tree
our_model = base.DecisionTree(criterion="information_gain", max_depth = optimum_depth)
our_model.fit(X_train, y_train)
our_y_pred = our_model.predict(X_test)
our_rmse = metrics.rmse(our_y_pred, y_test)

# Using sklearn decision tree
sklearn_model = DecisionTreeRegressor(max_depth = optimum_depth)
sklearn_model.fit(X_train, y_train)
sklearn_y_pred = sklearn_model.predict(X_test)
sklearn_rmse = metrics.rmse(sklearn_y_pred, y_test)

print("Optimal Depth:", optimum_depth)
print("RMSE for our decision tree:", our_rmse)
print("RMSE for sklearn decision tree:", sklearn_rmse)

"""
Optimal Depth: 4
RMSE for our decision tree: 11.072494355044617
RMSE for sklearn decision tree: 6.913069937096257
"""