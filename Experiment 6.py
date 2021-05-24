import pandas as pd
import numpy as np
data = pd.read_csv("svm.csv")

print(data['Class'].value_counts())
Malignant_dataset=data[data['Class']==4][0:200]
Benign_dataset=data[data['Class']==2][0:200]

data=data[pd.to_numeric(data['BareNac'], errors='coerce').notnull()]
data['BareNac']=data['BareNac'].astype('int')

feature_set=data[['Clump', 'Unifsize', 'Unishape', 'Margadh', 'Singleepisize', 'BareNac','Bland Chrom', 'Norm Nuc', 'Mit']]
x=np.asarray(feature_set)
y=np.asarray(data['Class'])
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

