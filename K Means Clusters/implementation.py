import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("K Means Clusters/income.csv")
# print(df)

scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

# plt.scatter(df.Age, df['Income($)'])
# plt.show()

sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)
plt.plot(k_rng, sse)
plt.xlabel('K')
plt.ylabel('Sum of Squasres Error')
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
# print(y_predicted)
df['Cluster'] = y_predicted
# print(df)

df1 = df[df.Cluster == 0]
df2 = df[df.Cluster == 1]
df3 = df[df.Cluster == 2]
plt.scatter(df1['Age'], df1['Income($)'], color='red')
plt.scatter(df2['Age'], df2['Income($)'], color='green')
plt.scatter(df3['Age'], df3['Income($)'], color='blue')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
# plt.legend()
plt.show()