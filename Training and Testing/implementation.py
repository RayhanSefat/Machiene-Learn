import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("Training and Testing/carprices.csv")
# print(df)

# plt.scatter(df['Mileage'], df['Sell Price($)'])
# plt.show()

# plt.scatter(df['Age(yrs)'], df['Sell Price($)'])
# plt.show()

X = df[['Mileage', 'Age(yrs)']]
# print(X)
y = df['Sell Price($)']
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# print(len(X_train))

clf = LinearRegression()
clf.fit(X_train, y_train)
print(y_test)
print(clf.predict(X_test))
print(clf.score(X_test, y_test))