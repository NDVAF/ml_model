from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from app.models import DecisionTreeClassifierParams
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model(ABC):
    @abstractmethod
    def train_and_predict(self, X_train, X_test, y_train):
        pass

class LogisticRegressionModel(Model):
    def train_and_predict(self, X_train, X_test, y_train):
       logger.info("Training Logistic Regression model")
       model = LogisticRegression(random_state=42)
       model.fit(X_train, y_train)
       y_pred_train = model.predict(X_train)
       y_pred_test = model.predict(X_test)
       logger.info("Logistic Regression model trained successfully")
       return y_pred_train, y_pred_test

class DecisionTreeClassifierModel(Model):
     def __init__(self, params: DecisionTreeClassifierParams = None):
        if params is None:
            self.params = DecisionTreeClassifierParams()
        else:
            self.params = params

     def train_and_predict(self, X_train, X_test, y_train):
        logger.info("Training DecisionTreeClassifier model")
        model = DecisionTreeClassifier(
            criterion=self.params.criterion,
            max_depth=self.params.max_depth,
            min_samples_leaf=self.params.min_samples_leaf,
            random_state=self.params.random_state
        )
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        logger.info("DecisionTreeClassifier model trained successfully")
        return y_pred_train, y_pred_test

class RandomForestModel(Model):
    def train_and_predict(self, X_train, X_test, y_train):
        logger.info("Training Random Forest model")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        logger.info("Random Forest model trained successfully")
        return y_pred_train, y_pred_test

class ModelFactory:
    def create_model(self, model_type: str, params: dict = None):
        logger.info(f"Creating model of type: {model_type}")
        if model_type == "logistic_regression":
            return LogisticRegressionModel()
        elif model_type == "random_forest":
            return RandomForestModel()
        elif model_type == "decision_tree":
            if params:
                 decision_tree_params = DecisionTreeClassifierParams(**params)
                 return DecisionTreeClassifierModel(decision_tree_params)
            else:
                return DecisionTreeClassifierModel()
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError("Unknown model type")