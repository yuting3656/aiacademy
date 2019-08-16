# random_forest


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
print("iris data shape: {}, iris target shape: {}".format(iris.data.shape, iris.target.shape))

random_seed = 5

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=random_seed)

rdf_clf = RandomForestClassifier(n_estimators=100)
rdf_clf.fit(x_train, y_train)

y_pred = rdf_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

