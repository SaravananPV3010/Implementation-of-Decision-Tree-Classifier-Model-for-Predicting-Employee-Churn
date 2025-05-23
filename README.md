# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```python
import pandas as pd
data=pd.read_csv("Employee.csv")
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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
## DATA HEAD
![image](https://github.com/user-attachments/assets/d8c0a80d-4ac5-4785-bba0-b92407612c93)

## DATA INFO
![image](https://github.com/user-attachments/assets/4daba4b1-8a6e-4408-b1b9-edf482edaed2)

## NULL DATA SET 
![image](https://github.com/user-attachments/assets/de423b4a-327c-46cb-9b94-027724d370f6)
## VALUE COUNT IN LEFT COLUMN 
TA![image](https://github.com/user-attachments/assets/24d3e944-23e6-4091-97a2-d5a1e5c06265)
## DATA PREDICTION
![image](https://github.com/user-attachments/assets/9e6e8150-8b58-4446-948d-7bfdf82160f3)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
