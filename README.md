# EX-6 Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values. 

5.From sklearn.model_selection import train_test_split. 

6.Assign the train dataset and test dataset. 

7.From sklearn.tree import DecisionTreeClassifier. 

8.Use criteria as entropy. 

9.From sklearn import metrics. 

10.Find the accuracy of our model and predict the require values.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: Dhivyapriya.R 

RegisterNumber: 212222230032  

```

```

import pandas as pd
data=pd.read_csv("/content/Employee (1).csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

## Data.head():

![image](https://github.com/dhivyapriyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477552/a83b0d32-c6ce-480a-8db4-ef9c6303f7e5)

## Data.info():

![image](https://github.com/dhivyapriyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477552/1b0ba252-a8cc-4d25-98c7-769b3517a6f7)

![image](https://github.com/dhivyapriyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477552/d1b9f977-da94-45ab-9d02-c2b3a6c26fe2)

## Data Value Counts():

![image](https://github.com/dhivyapriyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477552/fa453f76-8a5f-4196-a89b-a1d04c8636ca)

## Data.head() for salary:

![image](https://github.com/dhivyapriyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477552/1907ae5d-04b6-4217-a4c3-e3a4db575b58)

## x.head():

![image](https://github.com/dhivyapriyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477552/0ceb5d67-627b-4304-bd43-244538c624ca)

## Accuracy Value:

![image](https://github.com/dhivyapriyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477552/46cd20a2-be15-46e7-86ff-791584ea27dc)

## Data Prediction:

![image](https://github.com/dhivyapriyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477552/5554f429-5f06-4b90-aa7b-982a3d5e8f59)

## Result:

Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
