import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Salary_Data.csv")

exp =data.iloc[:,:-1].values
salary =data.iloc[:,1].values

from sklearn.model_selection import train_test_split
exp_train, exp_test, salary_train, salary_test = train_test_split(exp, salary, test_size=.2, random_state=1)
print(exp_train,"\n")
print(exp_test,"\n")
print(salary_train,"\n")
print(salary_test,"\n")

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(exp_train,salary_train)

salary_pred=regression.predict(exp_test)

print(salary_pred)
print("\n",salary_train)

plt.scatter(exp_train,salary_train, color='red')
plt.plot(exp_train,regression.predict(exp_train), color='blue')
plt.title('Salary Vs Experience(Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('salary')
plt.show()

plt.scatter(exp_test,salary_test, color='red')
plt.plot(exp_test,regression.predict(exp_test), color='blue')
plt.title('Salary Vs Experience(Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('salary')
plt.show()
