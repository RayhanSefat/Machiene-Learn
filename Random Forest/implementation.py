import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), df.target, test_size=0.2, random_state=65);

model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print(y_test)
print(y_predicted)
print("Score:", model.score(X_test, y_test))

cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()