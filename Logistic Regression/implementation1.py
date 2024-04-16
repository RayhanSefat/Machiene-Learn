import pandas as pd
from matplotlib import pyplot as plt
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv('Logistic Regression/HR_comma_sep.csv')
# print(df)

sal_df = df.groupby('salary').agg(salary_freq=('salary', 'count'), left_sum=('left', 'sum')).reset_index()
sal_df['retention_rate'] = 1 - sal_df['left_sum'] / sal_df['salary_freq']
# print(sal_df)
plt.bar(sal_df.salary, sal_df.salary_freq, color='green', width=0.4)
plt.bar(sal_df.salary, sal_df.left_sum, color='red', width=0.4)
plt.show()

dept_df = df.groupby('Department').agg(number_of_employees=('Department', 'count'), left_employees=('left', 'sum')).reset_index()
# print(dept_df)
plt.bar(dept_df.Department, dept_df.number_of_employees, color='green', width=0.4)
plt.bar(dept_df.Department, dept_df.left_employees, color='red', width=0.4)
plt.show()

