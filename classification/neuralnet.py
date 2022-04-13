from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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


X = [];
y = [];
for r in matrix[1:]:
    X.append(r[:-1])
    y.append(r[-1]);
X = numpy.array(X);
y = numpy.array(y);

FeatureSelection = ExtraTreesClassifier(n_estimators = 20)
FeatureSelection = FeatureSelection.fit(X,y)
FeatureSelection.feature_importances_
TheModel = SelectFromModel(FeatureSelection,prefit = True)
X_new = TheModel.transform(X)
X = X_new;

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

clf = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(100, 50));
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
# The accuracy of this model is: 89.2018779342723 %
# [[337   6   4]
#  [ 23  22   4]
#  [  6   3  21]]