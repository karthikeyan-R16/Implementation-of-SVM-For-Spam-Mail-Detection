# EXN0-9 Implementation-of-SVM-For-Spam-Mail-Detection
## Date : 
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by : Krthikeyan R
RegisterNumber : 212222240045
```
```
import chardet 
file = "C:/Users/SEC/Desktop/spam.csv"
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('C:/Users/SEC/Desktop/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### data.head()

![image](https://github.com/Ashwinkumar-03/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118663725/4a99c4df-975b-4ba3-a285-b783370f8ae8)

## data.info()

![image](https://github.com/Ashwinkumar-03/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118663725/7def7557-2f37-48e5-8c50-019a6755bb72)

## data.isnull().sum()

![image](https://github.com/Ashwinkumar-03/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118663725/83dad644-194c-4338-b72b-eaf91afd57a1)

## Y_prediction value

![image](https://github.com/bharathganeshsivasankaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119478098/ce982419-af68-49ac-bdfe-44bcc47d26c3)

## Accuracy value

![image](https://github.com/bharathganeshsivasankaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119478098/df4c4158-555f-4cfb-b0e1-a5dfc16d9727)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
