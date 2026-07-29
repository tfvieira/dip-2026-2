from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ImageClassifier:
    def __init__(self):
        self.knn = KNeighborsClassifier()
        self.svm = SVC()
        self.tree = DecisionTreeClassifier()
        self.bayes = GaussianNB()
        self.logistic = LogisticRegression(max_iter=1000)
        self.neural_net = MLPClassifier(max_iter=1000)

    def fit(self, X, y):
        """
        Fits all classifiers to the provided data.
        """
        self.knn.fit(X, y)
        self.svm.fit(X, y)
        self.tree.fit(X, y)
        self.bayes.fit(X, y)
        self.logistic.fit(X, y)
        self.neural_net.fit(X, y)

    def predict(self, X):
        """
        Predicts using all classifiers.
        """
        predictions = {
            "knn": self.knn.predict(X),
            "svm": self.svm.predict(X),
            "tree": self.tree.predict(X),
            "bayes": self.bayes.predict(X),
            "logistic": self.logistic.predict(X),
            "neural_net": self.neural_net.predict(X),
        }
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluates all classifiers and returns their accuracy.
        """
        predictions = self.predict(X_test)
        accuracy = {
            name: accuracy_score(y_test, pred) for name, pred in predictions.items()
        }
        return accuracy
