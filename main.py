import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('vgsales.csv')
X = df.drop(columns=['Year'])
y = df['Year']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# predictions = model.predict([21, 1], [22, 0])
predictions = model.predict(X_test)

score = accurancy_score(y_test, prediction)
score
# predictions
# df