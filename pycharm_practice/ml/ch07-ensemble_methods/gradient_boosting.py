from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = load_iris()


x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)


gb_clf = GradientBoostingClassifier(
    criterion='friedman_mse',
    n_estimators=100, # 生成的樹總數字
    learning_rate=0.1, # shrinkage of
    max_depth= 3,
    max_features= "auto"
)
gb_clf.fit(x_train, y_train)

y_pred = gb_clf.predict(x_test)

print(accuracy_score(y_test, y_pred))


