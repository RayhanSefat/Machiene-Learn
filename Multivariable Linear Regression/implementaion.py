import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import plotly.express as px
import math

df = pd.read_csv("E:/Machiene Learning/Multivariable Linear Regression/homeprice.csv")
# print(df)

median_bedroom = math.floor(df.bedrooms.median())
# print(median_bedroom)
df.bedrooms = df.bedrooms.fillna(median_bedroom)
print(df)