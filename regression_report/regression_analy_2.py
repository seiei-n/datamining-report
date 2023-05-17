import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
# import StandardScaler
from sklearn.preprocessing import StandardScaler


with open("boston.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')
# [0,1,3,4,5,7,8,9,10,11,12]
# Using feature 0 ,9
inputs = data[:, :13]
outputs = data[:, 13]

normalizer = StandardScaler()
inputs = normalizer.fit_transform(inputs)
coefs_ridge = []
r2_scores_ridge = []
coefs_lasso = []
r2_scores_lasso = []

for i in range(5000):

    # split data into training and testing
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        inputs, outputs, train_size=0.8)

    train_inputs = np.delete(train_inputs, [2, 6], 1)
    test_inputs = np.delete(test_inputs, [2, 6], 1)
    ridge = linear_model.Ridge(alpha=0.8)
    ridge.fit(train_inputs, train_outputs)
    coefs_ridge.append(ridge.coef_)
    r2_scores_ridge.append(ridge.score(test_inputs, test_outputs))


    lasso = linear_model.Lasso(0.001)
    lasso.fit(train_inputs, train_outputs)
    coefs_lasso.append(lasso.coef_)
    r2_scores_lasso.append(lasso.score(test_inputs, test_outputs))

def sort_coef(coef, names):
    abs_coef = np.abs(coef)
    abs_coef_sorted = np.sort(abs_coef)[::-1]
    # 降順にソートした時のインデックスを取得
    abs_coef_index_sorted = np.argsort(abs_coef)[::-1]
    # 降順にソートした時のデータ名を取得
    abs_coef_name_sorted = np.array(names)[abs_coef_index_sorted]
    # 降順にソートした時のデータ名と係数を表にする
    for name, coef in zip(abs_coef_name_sorted, abs_coef_sorted):
        # print like csv
        print(name, coef, sep=',')



data_names = ['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS',
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
ave_ridge = np.mean(coefs_ridge, axis=0)
ave_lasso = np.mean(coefs_lasso, axis=0)
ave_r2_ridge = np.mean(r2_scores_ridge)
ave_r2_lasso = np.mean(r2_scores_lasso)
sort_coef(ave_ridge, data_names)
print('R2 score', ave_r2_ridge, sep=',')
print('------------------')
sort_coef(ave_lasso, data_names)
print('R2 score', ave_r2_lasso, sep=',')





# # plot the graph
# plt.plot(ridge.predict(test_inputs), linestyle='dotted', color='red',label='ridge')
# plt.plot(lasso.predict(test_inputs), linestyle='dashed', color='blue',label='lasso')
# plt.plot(test_outputs, linestyle='solid', color='green',label='test')
# plt.legend()
# plt.show()


