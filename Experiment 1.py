#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#Importing dataset
dataset1 = pd.read_csv('/Users/Rohit/Desktop/Dataset.csv')

'''#X= Independent variables
X=dataset1.iloc[:,:-1].values#locate indexes, all the rows, -1 means the last column

#Y= dependent variables
Y=dataset1.iloc[:,-1].values #we will not give range, 

france	44.0	72000.0
spain	27.0	48000.0
germany	30.0	54000.0
spain 	38.0	51000.0
germany 	40.0	nan
france	35.0	58000.0
spain	nan	52000.0
france 	48.0	60000.0
germany	50.0	67000.0
france	37.0	83000.0

no
Yes
no
no
Yes
Yes
no
Yes
no
Yes'''