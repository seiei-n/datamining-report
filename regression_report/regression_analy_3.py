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

inputs = normalizer.fit_transform(inputs)



# split data into training and testing
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
    inputs, outputs, train_size=0.8, random_state=3)
poly = PolynomialFeatures(degree=3)
train_inputs = poly.fit_transform(train_inputs)
test_inputs = poly.fit_transform(test_inputs)
ridge = linear_model.Ridge(alpha=1)
ridge.fit(train_inputs, train_outputs)
print('Coefficients: \n', ridge.coef_)
# print('Intercept: \n', ridge.intercept_)
ridge_r2_score = ridge.score(test_inputs, test_outputs)


lasso = linear_model.Lasso(alpha=0.075)
lasso.fit(train_inputs, train_outputs)
print('Coefficients: \n', lasso.coef_)
# print('Intercept: \n', lasso.intercept_)
lasso_r2_score = lasso.score(test_inputs, test_outputs)

print('Ridge R2 score: \n', ridge_r2_score)
print('Lasso R2 score: \n', lasso_r2_score)


# plot the graph
plt.plot(ridge.predict(test_inputs), linestyle='solid', color='red',label='ridge')
plt.plot(lasso.predict(test_inputs), linestyle='dashed', color='blue',label='lasso')
plt.plot(test_outputs, linestyle='dotted', color='green',label='test')
plt.xlabel('Number of data')
plt.ylabel('Price')
plt.legend()
plt.show()
