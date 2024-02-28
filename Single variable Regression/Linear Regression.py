import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import plotly.express as px
import pickle
import joblib
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("Single variable Regression/homeprice.csv")

# fig = px.scatter(df, x = 'area', y = 'price', title='area vs price')
# fig.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# print(reg.coef_)
# print(reg.intercept_)
# print(reg.predict([[3350]]))

d = pd.read_csv("Single variable Regression/areas.csv")
# print(d)
p = reg.predict(d)
d['price'] = p
# print(d)
d.to_csv("Single variable Regression\prediction.csv", index=False)

# Plotting the points
plt.scatter(df['area'], df['price'], color='blue', label='Data Points')

# Plotting the regression line
plt.plot(df['area'], reg.predict(df[['area']]), color='red', label='Regression Line')

# Adding labels and title
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price with Regression Line')

# Adding legend
plt.legend()

# Show plot
plt.show()

with open('Single variable Regression/pickle_model', 'wb') as f:
    pickle.dump(reg, f)

joblib.dump(reg, 'Single variable Regression/joblib_model')