import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("Logistic Regression/insurance_data.csv")
# print(df)

# plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.2, random_state=5)
# print(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
print(X_test)
print(model.predict(X_test))
print(model.score(X_test, y_test))
print(model.predict_proba(X_test))