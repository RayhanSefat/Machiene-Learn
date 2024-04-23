import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
# print(df)

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
# plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')
# plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'],color="red",marker='*')
# plt.show()

# plt.xlabel('Petal Length')
# plt.ylabel('Petal Width')
# plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
# plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')
# plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'],color="red",marker='*')
# plt.show()

X = df.drop(['target'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
# print(knn.predict([[4.8,3.0,1.5,0.3]]))

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(classification_report(y_test, y_pred))