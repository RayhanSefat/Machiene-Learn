from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

rand_state = 111;

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3, random_state=rand_state)

def getModelScore(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# lr = LogisticRegression(solver='liblinear',multi_class='ovr')
# lr.fit(X_train, y_train)
# print("Logistic Regression Score", lr.score(X_test, y_test))
print("Logistic Regression Score", getModelScore(LogisticRegression(), X_train, X_test, y_train, y_test))

# svm = SVC(gamma='auto')
# svm.fit(X_train, y_train)
# print("SVM Score:", svm.score(X_test, y_test))
print("SVM Score:", getModelScore(SVC(), X_train, X_test, y_train, y_test))

# rf = RandomForestClassifier(n_estimators=40)
# rf.fit(X_train, y_train)
# print("Random Forest Score:", rf.score(X_test, y_test))
print("Random Forest Score:", getModelScore(RandomForestClassifier(), X_train, X_test, y_train, y_test))

kf = KFold(n_splits=3)
for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(train_index, test_index)

# folds = StratifiedKFold(n_splits=3)
# score_l = []
# score_svm = []
# score_rf = []
# for train_index, test_index in folds.split(digits.data, digits.target):
#     X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
#                                         digits.target[train_index], digits.target[test_index]
#     score_l.append(getModelScore(LogisticRegression(), X_train, X_test, y_train, y_test))
#     score_svm.append(getModelScore(SVC(), X_train, X_test, y_train, y_test))
#     score_rf.append(getModelScore(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))
# print(score_l)
# print(score_svm)
# print(score_rf)

print(cross_val_score(LogisticRegression(), digits.data, digits.target))
print(cross_val_score(SVC(), digits.data, digits.target))
print(cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target))