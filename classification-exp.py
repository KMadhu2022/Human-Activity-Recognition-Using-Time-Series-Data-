import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from tree import base
import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Write the code for Q2 a) and b) below. Show your results.

X_data = pd.DataFrame(X)
y_data = pd.DataFrame(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)
data = pd.DataFrame(X_train)

clf = base.DecisionTree(criterion="information_gain")
clf.fit(data,pd.Series(y_train))
y_pred = clf.predict(pd.DataFrame(X_test))

print("Accuracy:",metrics.accuracy(y_test,y_pred))
for cls in X_data.columns:
    print("Precision:", metrics.precision(y_test,y_pred,cls))
    print("Recall:", metrics.recall(y_test,y_pred,cls))

X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)

# Hyperparameters
k = 5 # Number of folds for cross validation
index = X_train.shape[0]//k
depths = tuple(range(1,11))
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

    model = base.DecisionTree(criterion = "information_gain", max_depth = depth)
    model.fit(X_train2, y_train2)
    y_pred = model.predict(X_validation)
    total_error += metrics.rmse(y_pred, y_validation)

  if total_error/k < min_error: # if error for this depth is min, set opt_depth = this depth
    optimum_depth = depth
    min_error = total_error/k

print()
print("Optimal Depth:", optimum_depth)

"""
Accuracy: 0.9666666666666667
Precision: 1.0
Recall: 0.9375
Precision: 0.9333333333333333
Recall: 1.0

Optimal Depth: 1
"""