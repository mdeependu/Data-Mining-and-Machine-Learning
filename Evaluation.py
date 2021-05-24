import pandas as pd
import numpy as np
data = pd.read_csv("Social_Network_Ads.csv")

print(data['Gender'].value_counts())
Purchased_Yes=data[data['Purchased']==1][0:100]
Purchased_No=data[data['Purchased']==0][0:100]

data=data[pd.to_numeric(data['Age'], errors='coerce').notnull()]
data['Age']=data['Age'].astype('int')

feature_set=data[['EstimatedSalary','Purchased']]
x=np.asarray(feature_set)
y=np.asarray(data['Purchased'])
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=.2,random_state=4)

from sklearn import svm
classifier= svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)

print(y_pred)
#evaluate the result
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

y_pred = classifier.predict(x_test)#Predicting the Test set results
age=dataset[dataset['Age']>=19][0:200]
EstimatedSalary=dataset[dataset['EstimatedSalary']>=20000][0:200]
axes= EstimatedSalar.plot(kind='scatter', x='Age',y='EstimatedSalary', color='Blue', label='purchase')
age.plot(kind='scatter', x='Age',y='EstimatedSalary', color='Red', label='Age', ax=axes)

#idenitifying unwanted dataset/ rows
dataset.dtypes

#Remove Unwanted rows: convert the value into numeric
dataset=dataset[pd.to_numeric(dataset['Gender'], errors='coerce').notnull()]
dataset.dtypes
dataset['Gender']=dataset['Gender'].astype('int')

dataset.columns
#Dependent type of attribute:
#Class
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))