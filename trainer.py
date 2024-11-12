import abc
from abc import abstractmethod
import os
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

class BaseModelTrainer(abc.ABC):
    """
    Abstract base class for model training.
    Defines methods for training, saving, deleting, and getting the status of a model.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None  
        self.status = "initialized" 

    @abstractmethod
    def train_model(self, X_train, y_train):
        """
        Trains the model on the provided training data.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
        Returns:
            str: Current metric of the trained model.
        """
        pass

    @abstractmethod
    def save_model(self, filepath):
        """
        Saves the trained model to the specified filepath.
        
        Args:
            filepath: Path where the model should be saved.
        """
        pass

    @abstractmethod
    def delete_saved_model(self, filepath):
        """
        Deletes the saved model from the specified filepath.
        
        Args:
            filepath: Path of the model to be deleted.
        """
        pass

    def get_status(self):
        """
        Returns the current status of the model (e.g., 'initialized', 'trained', 'saved').
        
        Returns:
            str: Current status of the model.
        """
        return self.status
    

class LogisticRegressionTrainer(BaseModelTrainer):
    """
    Concrete implementation of BaseModelTrainer for Logistic Regression.
    """

    def __init__(self, model_name, **kwargs):
        super().__init__(model_name)
        self.model = LogisticRegression(**kwargs)  
        
    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,\                                                                                           random_state=42)

    def train_model(self, data_path):
        """
        Trains the Logistic Regression model.
        """
        if data_path == '':
            self.setUp()
        else:
            df = pd.read_parquet(data_path)
            features = df.columns
            features.remove('target')
            X, y = df[features], df['target']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,\                                                                                           random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.status = "trained"
        return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    def save_model(self, filepath):
        """
        Saves the trained Logistic Regression model using joblib.
        """
        joblib.dump(self.model, filepath)
        self.status = "saved"
            
    def delete_saved_model(self, filepath):
        """
        Deletes the saved model file.
        """
        if os.path.exists(filepath):
            os.remove(filepath)
            self.status = "initialized" 
            
