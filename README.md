# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries.
2.Read the csv file.
3.Make necessary preprocessing.
4.Convert all the object data type to categorical data.
5.Now convert the categorial data to numbers representation using cat.codes
6.Define the feature and target variable.
7.Split the data as training and testing data.
8.Train the model using LogisticRegression().
9.Predict and test the accuracy of the model.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mourya G
RegisterNumber:  212224230170
*/
```
~~~
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  
  data=pd.read_csv("/content/Placement_Data.csv")
  data.info()
  
  data=data.drop('sl_no',axis=1)
  data
  
  data["gender"]=data["gender"].astype('category')
  data["ssc_b"]=data["ssc_b"].astype('category')
  data["hsc_b"]=data["hsc_b"].astype('category')
  data["hsc_s"]=data["hsc_s"].astype('category')
  data["degree_t"]=data["degree_t"].astype('category')
  data["workex"]=data["workex"].astype('category')
  data["specialisation"]=data["specialisation"].astype('category')
  data["status"]=data["status"].astype('category')
  data.dtypes
  
  data["gender"]=data["gender"].cat.codes
  data["ssc_b"]=data["ssc_b"].cat.codes
  data["hsc_b"]=data["hsc_b"].cat.codes
  data["hsc_s"]=data["hsc_s"].cat.codes
  data["degree_t"]=data["degree_t"].cat.codes
  data["workex"]=data["workex"].cat.codes
  data["specialisation"]=data["specialisation"].cat.codes
  data["status"]=data["status"].cat.codes
  data
  
  data=data.drop(['salary'],axis=1)
  data
  
  x=data.iloc[:,:-1].values
  y=data.iloc[:,-1]
  
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  
  from sklearn.linear_model import LogisticRegression
  model=LogisticRegression(max_iter=1000)
  model.fit(x_train,y_train)
  y_pred=model.predict(x_test)
  y_pred
  
  from sklearn.metrics import accuracy_score
  acc=accuracy_score(y_pred,y_test)
  acc
  
  model.predict([[0,85,0,92,0,2,75,2,1,1,0,0]])
~~~

## Output:
![427091082-c22e8b05-22fe-42c6-8487-5281b7572db7](https://github.com/user-attachments/assets/0e532386-f6dd-49a9-84ba-357c7cad09fa)
![427091299-92e4782a-2310-4d38-8811-7cefd4d1dea4](https://github.com/user-attachments/assets/80807d29-9ad7-4dc3-a50c-5153f42d81f6)
![427091394-cfb58215-90f3-4f20-b002-c49d0ac04f73](https://github.com/user-attachments/assets/3c912bd9-815d-4d4e-a0c3-8ec2fae26644)
![427091486-4a7bf9ae-0b21-4c21-8db0-e4d50227f26d](https://github.com/user-attachments/assets/86de8d5c-7521-4299-b361-36a6cfde4053)
![427091620-b87444d0-8cd1-40e4-a8f1-faf910b72949](https://github.com/user-attachments/assets/bd153319-da65-4242-96ff-61842931cf6d)
![427091728-463ae37f-7908-43b1-a7f2-55613bbd9912](https://github.com/user-attachments/assets/97e9f61a-2d58-4beb-96ec-c8d336a04a34)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
