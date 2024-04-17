import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("Decision Tree/salaries.csv")
# print(df)

inputs = df.drop('salary_more_then_100k', axis='columns')
target = df.salary_more_then_100k
# print(inputs)
# print(target)

le = LabelEncoder()
inputs['company_n'] = le.fit_transform(inputs['company'])
inputs['job_n'] = le.fit_transform(inputs['job'])
inputs['degree_n'] = le.fit_transform(inputs['degree'])
# print(inputs)
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')
# print(inputs_n)

merged = pd.concat([inputs_n, target], axis='columns')

X_train, X_test, y_train, y_test = train_test_split(merged[['company_n', 'job_n', 'degree_n']], merged.salary_more_then_100k, test_size=0.2, random_state=121)

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
print(X_test)
print(y_test)
print(model.predict(X_test))
print("Score:" ,model.score(X_test, y_test))