# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Deependu Mandal\Desktop\iphone_purchase_records.csv')

# All the indepent variables into X variable
X = dataset.iloc[:,0:3].values

# All dependent in y variable.
y = dataset.iloc[:, -1].values

# Changing all the catagorical values into integers
from sklearn.preprocessing import LabelEncoder as le
label_encoder_gender= le()
X[:,0] = label_encoder_gender.fit_transform(X[:,0])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the classification report
from sklearn.metrics import classification_report

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix


"LOGICAL REGRESSION"

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0,solver="liblinear")
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier1.predict(X_test)

print("________________________________________________________________________")
print("LOGISTIC REGRESSION")
print(classification_report(y_test, y_pred1))
cm1= confusion_matrix(y_test, y_pred1)
print(cm1)


"K-NEAREST NEIGHBOURS"

# Training the KNN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)

print("________________________________________________________________________")
print("K-NEAREST NEIGHBOURS")
print(classification_report(y_test, y_pred2))
cm2= confusion_matrix(y_test, y_pred2)
print(cm2)


"Naive Bayes"

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit(X_train, y_train)

# Predicting the Test set results
y_pred3 = classifier3.predict(X_test)

print("________________________________________________________________________")
print("NAIVE BAYES")
print(classification_report(y_test, y_pred3))
cm3= confusion_matrix(y_test, y_pred3)
print(cm3)


"SUPPORT VECTOR MACHINE"

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf', random_state = 0)
classifier4.fit(X_train, y_train)

# Predicting the Test set results
y_pred4 = classifier4.predict(X_test)

print("________________________________________________________________________")
print("SUPPORT VECTOR MACHINE")
print(classification_report(y_test, y_pred4))
cm4= confusion_matrix(y_test, y_pred4)
print(cm4)


"DECISION TREE"

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier5 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier5.fit(X_train, y_train)

# Predicting the Test set results
y_pred5 = classifier5.predict(X_test)

print("________________________________________________________________________")
print("DECISION TREE")
print(classification_report(y_test, y_pred5))
cm5= confusion_matrix(y_test, y_pred5)
print(cm5)