import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("D:\Data mining\Practical\DECISION TREE.csv")
train , test  = train_test_split(data , test_size=0.15)

Pos_tempo=dataset[dataset['target']==1]['tempo']
Neg_tempo=dataset[dataset['target']==0]['tempo']
fig=plt.figure(figsize=(12,8))
plt.title("SONG TEMPO LIKE/ DISLIKE DISTRIBUTION")
Pos_tempo.hist(alpha=.7,bins=30, label="POSTIVE")
Neg_tempo.hist(alpha=.7,bins=30, label="NEGATIVE")
plt.legend(loc="upper right")
plt.plot()
plt.show()

c = DecisionTreeClassifier(min_samples_split=100)
features=['danceability','loudness','valence','energy','instrumentalness','acousticness','key','speechiness','duration_ms']
x_train = train[features]
y_train = train['target']
x_test = test[features]
y_test = test["target"]
dt = c.fit(x_train , y_train)

feature_name = list(x_train.columns)
target_name = ['target']
class_name = list(data['target'])
classes = []
for x in class_name:
    classes.append(str(x))

fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(c,
                   feature_names=feature_name,
                   class_names=classes,
                   filled=True)
fig.savefig("practice.png")

