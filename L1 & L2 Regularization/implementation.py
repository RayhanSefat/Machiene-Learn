import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("L1 & L2 Regularization/Melbourne_housing_FULL.csv")
# print(df)
# print(df.nunique())
# print(df.shape)

cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
df = df[cols_to_use]

cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)
df['Landsize'] = df['Landsize'].fillna(df.Landsize.mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df.BuildingArea.mean())
df.dropna(inplace=True)
# print(df.shape)

# print(df)

df = pd.get_dummies(df, drop_first=True)
# print(df)

X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))
print(reg.score(X_train, y_train))

lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(X_train, y_train)
print(lasso_reg.score(X_test, y_test))
print(lasso_reg.score(X_train, y_train))

ridge_reg= linear_model.Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(X_train, y_train)
print(ridge_reg.score(X_test, y_test))
print(ridge_reg.score(X_train, y_train))