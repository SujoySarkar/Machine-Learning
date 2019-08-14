import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# loading the data set
data = pd.read_csv('p.csv')
# plot the data
X = data['Size'].values.reshape(-1, 1)
Y = data['Price'].values.reshape(-1, 1)
plt.plot(X, Y, 'o', color='g')
# plt.show()
# selecting linear regression model
X_train, Y_train = X[0:3], Y[0:3]
X_test, Y_test = X[3:], Y[3:]
# print(X_train, X_test)
Model = LinearRegression()
Model.fit(X_train, Y_train)
regression_line = Model.predict(X)

plt.plot(X, regression_line)
plt.plot(X_train, Y_train, 'o')
plt.plot(X_test, Y_test, 'o')
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Pizza price prediction')
a = Model.predict([[7]])
print(a)
# plt.show()
# y_prediction = Model.predict(X_test)
# print(y_prediction)
# a = r2_score(y_true=Y_test, y_pred=y_prediction)
# g = Model.score(X_test, Y_test)
# print(g)



