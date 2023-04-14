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

# persisting models
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, 'music-recommender.joblib')

# predictions = model.predict([21, 1])

# -------------------------------------
# visualizing decision trees
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

tree.export_graphviz(model, out_file='music_recommender.dot', 
                    feature_names=['age', 'gender'], 
                    class_names=sorted(y.unique())
                    label='all', 
                    rounded=True,
                    filled=True)
