from sklearn.datasets import load_digits
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

dataset = load_digits()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# print(df)

X = df
y = dataset.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Score before PCA:", model.score(X_test, y_test))

pca = PCA(0.95)     # retain 95% of the useful features and then create new dimentions
X_pca = pca.fit_transform(X)
print(X_pca.shape)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print("Score after PCA:", model.score(X_test_pca, y_test))

pca = PCA(n_components=5)   # 5 most significant dimesions only
X_pca = pca.fit_transform(X)
print(X_pca.shape)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print("Score with 5 most significant dimesions:", model.score(X_test_pca, y_test))