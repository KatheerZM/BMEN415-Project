import math
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

with open ('../Volumetric_features.csv', mode='r') as volumetric:
    text = volumetric.read();

line_split = text.split("\n");

matrix = [];
matrix.append(line_split[0].split(','));

for line in line_split[1:-1]:
    comma_split = line.split(",");
    for n in range(len(comma_split)):
        comma_split[n] = float(comma_split[n]);
    matrix.append(comma_split);

#print(matrix[0:10])


# test classification dataset
from collections import Counter
from sklearn.datasets import make_classification
import sklearn.linear_model
# define dataset

X = [];
y = [];
for r in matrix[1:]:
    X.append(r[:-2])
    y.append(r[-2]);
X = numpy.array(X);
y = numpy.array(y);
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1);
print(X.shape, y.shape)
print(Counter(y))

sc = StandardScaler()
xTrain = sc.fit_transform(X_train) #fits training data to rest
xTest = sc.transform(X_test) #scales data

# solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 8, 6, 4, 2),
clf = MLPRegressor(random_state=1, max_iter=2000);

sgd = make_pipeline(StandardScaler(),clf)


sgd.fit(X_train, Y_train);

Y_predict = sgd.predict(X_test)

print ("R^2 score:", r2_score(Y_test, Y_predict))
print("Mean Squared Error:", mean_squared_error(Y_test, Y_predict))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_test, Y_predict))


# Result:
#
# R^2 score: 0.8439345243362766
# Mean Squared Error: 62.168481264642445
# Mean Absolute Percentage Error: 0.12319018948791459

