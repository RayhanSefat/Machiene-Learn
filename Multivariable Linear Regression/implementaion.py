import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import plotly.express as px
import math
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("Multivariable Linear Regression\homeprice.csv")
# print(df)

median_bedroom = math.floor(df.bedrooms.median())
# print(median_bedroom)
df.bedrooms = df.bedrooms.fillna(median_bedroom)
# print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
# print(reg.coef_)
# print(reg.intercept_)

print(reg.predict([[3000, 3, 40]]))
print(reg.predict([[2500, 4, 5]]))