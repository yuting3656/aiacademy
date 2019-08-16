from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

# random_seed
random_seed = 5
X_train, x_test, Y_train, y_test = train_test_split(iris.data, iris.target, random_state=random_seed)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)

y_pred = dt_clf.predict(x_test)
print(y_pred)
print(x_test)

print(accuracy_score(y_test, y_pred))
