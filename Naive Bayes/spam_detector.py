import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

df = pd.read_csv("Naive Bayes/spam.csv")
df['Spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
# print(df)

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.2, random_state=125)

v = CountVectorizer()
X_train_count = v.fit_transform(X_train)
X_test_count = v.transform(X_test)
# print(X_train_count.toarray()[0:5])

model = MultinomialNB()
model.fit(X_train_count, y_train)
print(model.score(X_test_count, y_test))

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
print(model.predict(emails_count))

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.predict(emails))