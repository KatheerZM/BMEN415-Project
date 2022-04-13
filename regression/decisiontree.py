import math

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from collections import Counter

with open ('../Volumetric_features.csv', mode='r') as volumetric:
    text = volumetric.read();
line_split = text.split("\n");
matrix = [];
matrix.append(line_split[0].split(','));
for line in line_split[1:1200]:
    comma_split = line.split(",");
    for n in range(len(comma_split)):
        comma_split[n] = float(comma_split[n]);
    matrix.append(comma_split);


X = [];
y = [];
for r in matrix[1:]:
    X.append(r[1:-2])
    y.append(r[-2]);
X = numpy.array(X);
y = numpy.array(y);
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
print(X.shape, y.shape)
print(Counter(y))

sc = StandardScaler()
xTrain = sc.fit_transform(X_train) #fits training data to rest
xTest = sc.transform(X_test) #scales data

clf = DecisionTreeRegressor(criterion='friedman_mse', min_weight_fraction_leaf=0.00);
clf = make_pipeline(StandardScaler(),clf)
clf.fit(X_train, Y_train);
Y_predict = clf.predict(X_test)

print ("R^2 score:", r2_score(Y_test, Y_predict))
print("Mean Squared Error:", mean_squared_error(Y_test, Y_predict))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_test, Y_predict))

#tree.plot_tree(sgd)
#plt.show()

XY = np.vstack([Y_predict,Y_test])
z = gaussian_kde(XY)(XY)
fig, ax = plt.subplots()
sns.regplot(Y_predict ,Y_test, fit_reg=True, scatter_kws={"s": 100})
ax.scatter(Y_predict,Y_test, c=z, s=100)
ax.set_xlabel('Predicted Age')
ax.set_ylabel('Actual Age')
plt.show()

# Result:
#
# R^2 score: 0.6787003979417512
# Mean Squared Error: 8.179166666666667
# Mean Absolute Percentage Error: 0.02417994736721368



