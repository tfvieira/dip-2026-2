from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        return score
