import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter


#Extract Data from csv
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

#Split data into X and y
X = [];
y = [];
for r in matrix[1:]:
    X.append(r[:-1])
    y.append(r[-1]);
X = numpy.array(X);
y = numpy.array(y);

#Select Features
FeatureSelection = ExtraTreesClassifier(n_estimators = 20)
FeatureSelection = FeatureSelection.fit(X,y)
FeatureSelection.feature_importances_
TheModel = SelectFromModel(FeatureSelection,prefit = True)
X_new = TheModel.transform(X)
X_new.shape;
X = X_new;

#Split X and y for testing and training
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

#Logistic Regression
model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_train, Y_train);
Y_predict = model.predict(X_test);

#Print Results
print('The accuracy of this model is:', model.score(X_test,Y_test)*100,'%')
LDA_classification = LinearDiscriminantAnalysis()
cm_LDA = confusion_matrix(Y_test,Y_predict)
print(cm_LDA)

#Plot Confusion Matrix
plt.figure(figsize=(6,4))
cm_LDA_df = pd.DataFrame(cm_LDA,index = ['1','2','3'], columns = ['1','2','3'])
sns.heatmap(cm_LDA_df, annot=True)
plt.xlabel("Predict")
plt.ylabel("True")
plt.show();


# Result:
#
# The accuracy of this model is: 89.2018779342723 %
# [[329   5   4]
#  [ 26  35   2]
#  [  5   4  16]]
