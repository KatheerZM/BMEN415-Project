import math
from sklearn.neural_network import MLPClassifier
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


#Logistic Regression

# define the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_train, Y_train);


Y_predict = model.predict(X_test);


print('The accuracy of this model is:', model.score(X_test,Y_test)*100,'%')
LDA_classification = LinearDiscriminantAnalysis()
cm_LDA = confusion_matrix(Y_test,Y_predict)
print(cm_LDA)
cm_LDA_df = pd.DataFrame(cm_LDA,index = ['1','2','3'], columns = ['1','2','3'])

plt.figure(figsize=(6,4))
sns.heatmap(cm_LDA_df, annot=True)
plt.xlabel("Predict")
plt.ylabel("True")
plt.show();




exit()

# define dataset
# X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=21, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)));

# Result:
#
# The accuracy of this model is: 87.3040752351097 %
# [[456  17   5]
#  [ 37  53   4]
#  [  7  11  48]]
