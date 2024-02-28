import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import plotly.express as px
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

def gradient_descent(x, y):
    m_curr = 1.0
    b_curr = 2.0
    iterations = 100
    n = len(x)
    learning_rate = 0.0001

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iteratin {}".format(m_curr, b_curr, cost, i))

df = pd.read_csv("Gradient Descent/marks.csv")
# print(df)

x = np.array(df['math'])
y = np.array(df['cs'])
# print(x, y)

gradient_descent(x, y)

reg = linear_model.LinearRegression()
reg.fit(df[['math']], df.cs)
print("m = {}, b = {}".format(reg.coef_, reg.intercept_))