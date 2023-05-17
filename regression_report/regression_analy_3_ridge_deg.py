import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
# import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

with open("boston.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')
normalizer = StandardScaler()
inputs = data[:, [4,5,7,10,12]]
outputs = data[:, 13]
r2_scores_ridge = []
inputs = normalizer.fit_transform(inputs)

for n in range(1, 9):
    r2_scores_ridge_for_average = []
    for i in range(100):
        # split data into training and testing
        train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
            inputs, outputs, train_size=0.8)
    
        poly = PolynomialFeatures(degree=n)
        train_inputs = poly.fit_transform(train_inputs)
        test_inputs = poly.fit_transform(test_inputs)
        ridge = linear_model.Ridge(alpha=1)
        ridge.fit(train_inputs, train_outputs)
        # print('Coefficients: \n', ridge.coef_)
        # print('Intercept: \n', ridge.intercept_)
        r2_score = ridge.score(test_inputs, test_outputs)
        r2_scores_ridge_for_average.append(r2_score)
        # print('R2 score: \n', r2_score)
    average_r2_score_ridge = sum(r2_scores_ridge_for_average) / len(r2_scores_ridge_for_average)
    r2_scores_ridge.append(average_r2_score_ridge)

max_r2_score_ridge = max(r2_scores_ridge)
print('max_r2_score_ridge: \n', max_r2_score_ridge)

# plot the graph
plt.plot(np.arange(1, 9), r2_scores_ridge)
plt.xlabel('degree')
plt.ylabel('R2 score')
plt.xlim(1, 5)
plt.ylim(0, 1)
plt.show()