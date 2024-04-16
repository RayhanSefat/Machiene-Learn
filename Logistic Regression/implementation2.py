from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=15)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

print(digits.target[100:105])
print(model.predict(digits.data[100:105]))

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)

plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()