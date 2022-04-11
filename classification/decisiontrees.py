import math
from sklearn.neural_network import MLPClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn import tree
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

with open ('../fetal_health.csv', mode='r') as fetal:
    text = fetal.read();

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
    X.append(r[:-1])
    y.append(r[-1]);
X = numpy.array(X);
y = numpy.array(y);
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1);
print(X.shape, y.shape)
print(Counter(y))
# solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 8, 6, 4, 2),
clf = tree.DecisionTreeClassifier();
clf.fit(X_train, Y_train);

Y_predict = clf.predict(X_test)

print('The accuracy of this model is:', clf.score(X_test,Y_test)*100,'%')
LDA_classification = LinearDiscriminantAnalysis()
cm_LDA = confusion_matrix(Y_test,Y_predict)
print(cm_LDA)
cm_LDA_df = pd.DataFrame(cm_LDA,index = ['1','2','3'], columns = ['1','2','3'])

plt.figure(figsize=(6,4))
sns.heatmap(cm_LDA_df, annot=True)
plt.xlabel("Predict")
plt.ylabel("True")
plt.show();

# Result:
#
# Counter({1.0: 1655, 2.0: 295, 3.0: 176})
# The accuracy of this model is: 92.16300940438872 %
# [[480  19   1]
#  [ 21  64   2]
#  [  6   1  44]]