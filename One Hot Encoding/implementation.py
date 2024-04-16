import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.express as px
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("One Hot Encoding/homeprice.csv")
# print(df)

dummies = pd.get_dummies(df.town)
# print(dummies)

merged = pd.concat([df, dummies], axis="columns")
# print(merged)

final = merged.drop(['town', 'west windsor'], axis="columns")
# print(final)

model = LinearRegression()
X = final.drop('price', axis="columns")
# print(X)
y = final.price
model.fit(X, y)
print(model.predict([[2800, 0, 1]]))
print(model.predict([[3400, 0, 0]]))
print(model.score(X, y))