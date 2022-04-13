import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

#Test Train split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #fits training data to rest
X_test = sc.fit_transform(X_test) #scales data

clf = MLPRegressor(solver='adam', max_iter=1000, hidden_layer_sizes=(150, 100));
clf.fit(X_train, Y_train);
Y_predict = clf.predict(X_test)

print ("R^2 score:", r2_score(Y_test, Y_predict))
print("Mean Squared Error:", mean_squared_error(Y_test, Y_predict))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_test, Y_predict))

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
# R^2 score: -0.12363444903928333
# Mean Squared Error: 24.2718891347673
# Mean Absolute Percentage Error: 0.05060599716580764

