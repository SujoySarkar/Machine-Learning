import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# load the data


df = pd.read_csv("student_scores.csv")

# visualization the data
X = df["Hours"].values.reshape(-1, 1)
Y = df["Scores"].values.reshape(-1, 1)
plt.plot(X, Y, 'o')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Students scores according to study")
# plt.show()

# taking training and testing data
x_train,y_train=X[0:20],Y[0:20]
x_test,y_test=X[20:],Y[20:]
# print(x_train,x_test)


# import linear regression from sklearn.linear_model
model=LinearRegression()
# fit attribute takes array,so i have write X = df["Hours"].values.reshape(-1, 1) also in Y
model.fit(X,Y)
# creating a regression line
regression_line=model.predict(X)
plt.plot(X,regression_line,color='g')
plt.plot(x_train,y_train,'o',color='red')
plt.plot(x_test,y_test,'o',color='yellow')
# plt.show()
y_prediction=model.predict(x_test)
print(mean_squared_error(y_test,y_prediction))
