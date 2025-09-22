import numpy as np
import pandas as pd
import pickle

data = pd.read_csv('Salary_Data.csv')

x = data['YearsExperience'].values
y = data['Salary'].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)

with open("salary_prediction_model","wb") as model_file:
    pickle.dump(reg,model_file)