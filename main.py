import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('vgsales.csv')

X = df.drop(columns=['Year'])
Y = df['Name']
model = DecisionTreeClassifier()
model.fit(X, Y)
model