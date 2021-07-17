from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = datasets.load_iris()
# print(iris)

X = iris.data
y = iris.target
# print(X)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y)

log_model = LogisticRegression()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()

log_model = log_model.fit(x_train, y_train)
knn_model = knn_model.fit(x_train, y_train)
rf_model = rf_model.fit(x_train, y_train)

pickle.dump(log_model, open('log_model.pkl', 'wb'))
pickle.dump(knn_model, open('knn_model.pkl', 'wb'))
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
