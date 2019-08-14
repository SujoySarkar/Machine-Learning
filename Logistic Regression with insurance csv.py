import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# loading the data set
data = pd.read_csv('dh.csv')
# x = data['age'].values
# y = data['bought_insurance'].values
# plt.scatter(x, y, marker='+', color='b')
# plt.show()
# split the data set for training and testing
x_train, x_test, y_train, y_test = train_test_split(data[['age']], data['bought_insurance'], test_size=0.1)
# print(x_train)
# creating the model
model = LogisticRegression()
model.fit(x_train, y_train)
print(x_test)
a = model.predict(x_test)
print(a)
b = model.predict_proba(x_test)
print(b)
# s = model.score(x_test, y_test)
# print(s)
c = model.predict([[11]])
print(c)




