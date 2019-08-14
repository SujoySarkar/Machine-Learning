import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Load the data


data=pd.read_csv("titanic.csv")
data=data.drop(['Name'],axis=1)
# print(data.columns)
# ploting the data
age=data["Age"].values
fare=data["Fare"].values
survived=data["Survived"].values


# if survive then color will show green otherwise red
colors=[]
for item in survived:
    if item==0:
        colors.append('red')
    else:
        colors.append('green')
#plt.scatter(age,fare,s=40,color=colors)
#plt.show()

# from data set removing Survived column and Target value has only survive data.
Feature=data.drop(['Survived'],axis=1).values

Target=data['Survived'].values
# selecting training data and test data
feature_train,target_train=Feature[0:710],Target[0:710]
feature_test,target_test=Feature[710:],Target[710:]


#creating the model
model=GaussianNB()
model.fit(feature_train,target_train)
predicted_value=(model.predict(feature_test))
for item in zip(target_test,predicted_value):
    print("Actual was : ",item[0],"predicted was : ",item[1])
print(model.score(feature_test,target_test))
