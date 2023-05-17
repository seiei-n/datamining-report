import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
# import StandardScaler
from sklearn.preprocessing import StandardScaler

with open("boston.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')
data_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',
              'tax', 'ptratio', 'b', 'lstat', 'medv']
# locate All rows and 0-12 columns
inputs = data[:, :13]
outputs = data[:, 13]
regr = linear_model.LinearRegression()
normalizer = StandardScaler()
inputs = normalizer.fit_transform(inputs)
scores_1 = []
coef_1 = []
for i in range(5000):
    # split data into training and testing
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        inputs, outputs, train_size=0.8,shuffle=True)
    
    regr.fit(train_inputs, train_outputs)
    # print('Coefficients: \n')
    # print coefficients and data_names at the same time
    # for name, coef in zip(data_names, regr.coef_):
    #     print(name, ': ', coef)  
    # print('Intercept: \n', regr.intercept_)
    coef_1.append(regr.coef_)
    r2_score_1 = regr.score(test_inputs, test_outputs)
    scores_1.append(r2_score_1)
    # print('R2 score: \n', r2_score_1)


average_coef_1 = np.mean(coef_1, axis=0)
print('average_coef_1: \n', average_coef_1)


# 絶対値
abs_coef = np.abs(average_coef_1)
# 降順にソート
abs_coef_sorted = np.sort(abs_coef)[::-1]
# 降順にソートした時のインデックスを取得
abs_coef_index_sorted = np.argsort(abs_coef)[::-1]
# 降順にソートした時のデータ名を取得
abs_coef_name_sorted = np.array(data_names)[abs_coef_index_sorted]
# 降順にソートした時のデータ名と係数を表にする
for name, coef in zip(abs_coef_name_sorted, abs_coef_sorted):
    print(name,',',coef)



print('\n')





train_inputs_2, test_inputs_2, train_outputs_2, test_outputs_2 = train_test_split(
        inputs, outputs, train_size=0.8)
r2_score_n = []
# 0.1から3まで0.1刻みでループ
r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1.1, 1.2,
     1.3, 1.4, 1.5, 1.6, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
for n in r:
    coef_abs_over_n = abs_coef[abs_coef > n]
    coef_abs_over_n_index = np.where(abs_coef > n)
    coef_abs_over_n_name = np.array(data_names)[coef_abs_over_n_index]
    scores_3 = []
    test_inputs_over_n = test_inputs_2[:, coef_abs_over_n_index[0]]
    train_inputs_over_n = train_inputs_2[:, coef_abs_over_n_index[0]]
    regr.fit(train_inputs_over_n, train_outputs_2)
    # print('Coefficients: \n')
    # # print coefficients and data_names at the same time
    # for name, coef in zip(coef_abs_over_1_name, regr.coef_):
    #     print(name, ': ', coef)
    # print('Intercept: \n', regr.intercept_)
    r2_score_3 = regr.score(test_inputs_over_n, test_outputs_2)
    r2_score_n.append(r2_score_3)



# 一番いいスコアを取得
r2_score_3 = np.max(r2_score_n)
# best n
B_n = r[np.argmax(r2_score_n)]

# B_n以上の係数を取得
coef_abs_over_n = abs_coef[abs_coef > B_n]
coef_abs_over_n_index = np.where(abs_coef > B_n)
coef_abs_over_n_name = np.array(data_names)[coef_abs_over_n_index]
# coef_abs_over_n と coef_abs_over_n_index と coef_abs_over_n_name を表示
for name, coef in zip(coef_abs_over_n_name, coef_abs_over_n):
    print(name, ': ', coef)
print(coef_abs_over_n_index[0])
regr2 = linear_model.LinearRegression()
regr2.fit(train_inputs_2, train_outputs_2)
r2_score_1 = regr2.score(test_inputs_2, test_outputs_2)



print('R2 score 1: \n', r2_score_1)
print('n: \n', B_n)
print('Best R2 score 3_mean: \n', r2_score_3)
train_inputs_over_n = train_inputs_2[:, coef_abs_over_n_index[0]]

plt.scatter(np.arange(len(test_outputs_2)), regr.predict(test_inputs_over_n), color='red')
plt.scatter(np.arange(len(test_outputs_2)), test_outputs_2, color='black')
plt.xlabel('data number')
plt.ylabel('price')

plt.show()


plt.xlabel('Number of data')
plt.ylabel('difference')
plt.plot(np.arange(len(test_outputs_2)), regr.predict(test_inputs_over_n) - test_outputs_2, color='red')
plt.scatter(np.arange(len(test_outputs_2)), regr.predict(test_inputs_over_n) - test_outputs_2, color='red')
plt.show()
    