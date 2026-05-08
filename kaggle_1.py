import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('3/Titanic-Dataset.csv')

df.drop(columns =['PassengerId','Name', 'Ticket','Cabin'], inplace=True)

mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

gender_map = {'male': 0,
              'female': 1}
df['Sex'] = df['Sex'].map(gender_map)

embarked_map = {'S': 0,
                'C': 1,
                'Q':2}
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]).map(embarked_map)

y = df['Survived']

X = np.column_stack([df['Pclass'], df['Sex'], df['Age'], df['SibSp'], df['Parch'], df['Fare'], df['Embarked']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

results = {}

for md in range(1, 20):
    clf = RandomForestClassifier(n_estimators = 500, random_state = 42, max_depth = md)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    results[md] = score
    

best_md = max(results, key=results.get)
print(results[best_md])

