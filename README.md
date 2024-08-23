# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 2.Set variables for assigning dataset values. 3.Import linear regression from sklearn. 4.Assign the points for representing in the graph. 5.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kannan.S
RegisterNumber: 212223230098
```
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
# Dataset:
![image](https://github.com/user-attachments/assets/afa6544d-af6f-408c-a0b1-15469b543fb9)

# Head values:
![image](https://github.com/user-attachments/assets/3667d117-15cc-48b1-9c96-cb0c602c947e)

# Tail values:
![image](https://github.com/user-attachments/assets/2381925e-3ff7-4e3a-a02c-9cd40e24fe6f)

# X and Y values:
![image](https://github.com/user-attachments/assets/71da57ea-b60b-4d19-96c8-c3fe67b1da4b)

# Predication values of X and Y:
![image](https://github.com/user-attachments/assets/32f351de-28b6-4d71-8708-64d7ce322c2b)

# MSE,MAE and RMSE:
![image](https://github.com/user-attachments/assets/f51642b0-66a8-411b-83be-d0a8b3924f00)

# Training Set:
![image](https://github.com/user-attachments/assets/cd3d5ce2-d122-4f61-869d-95d4606d20ec)

# Testing Set:
![image](https://github.com/user-attachments/assets/6a26960f-ab3e-4d2e-ba44-718c89cc644c)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
