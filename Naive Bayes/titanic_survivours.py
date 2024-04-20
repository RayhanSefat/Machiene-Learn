import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("Naive Bayes/titanic.csv")
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
# print(df)

target = df.Survived
inputs = df.drop(['Survived'], axis='columns')
dummies = pd.get_dummies(inputs.Sex)
# print(dummies)
inputs = pd.concat([inputs, dummies], axis='columns')
inputs = inputs.drop('Sex', axis='columns')
# print(inputs)
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=121)

model = GaussianNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))