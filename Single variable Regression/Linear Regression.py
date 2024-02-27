import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import plotly.express as px

df = pd.read_csv("homeprice.csv")

# fig = px.scatter(df, x = 'area', y = 'price', title='area vs price')
# fig.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# print(reg.coef_)
# print(reg.intercept_)
# print(reg.predict([[3350]]))

d = pd.read_csv("areas.csv")
# print(d)
p = reg.predict(d)
d['price'] = p
# print(d)
d.to_csv("prediction.csv", index=False)

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