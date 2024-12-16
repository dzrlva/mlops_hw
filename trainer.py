import abc
from abc import abstractmethod
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

class BaseModelTrainer(abc.ABC):
    """
    Abstract base class for model training.
    Defines methods for training, saving, deleting, and getting the status of a model.
    """

    def __init__(self, logs_path):
        """
            Initialize model.
        """
        pass

    @abstractmethod
    def train_model(self):
        """
        Trains the model on the provided training data.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
        Returns:
            str: Current metric of the trained model.
        """
        pass
    
    def test_model(self):
        """
        Test the model on the provided test data.
        
        Args:
            X_test: Testing features.
            y_test: Testing labels.
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
    

class LogisticRegressionTrainer(BaseModelTrainer):
    """
    Implementation of BaseModelTrainer for Logistic Regression.
    """

    def __init__(self, model=None, model_id=None, logs_path='./logs_book.txt'):
        if model:
            self.model = model
            self.logs_book = logs_path
            self.status = "model initialized"
            self.id = model_id
        else:
            self.model = LogisticRegression()
            self.status = "model created"
            self.logs_book = logs_path
            with open(self.logs_book, 'r') as f:
                logs = f.read()
            self.id = len(set([num.split(' ')[0] for num in logs.split('\n')]))
            self.add_log()

    def get_model_id(self):
        return self.id
    
    def add_log(self):
        with open(self.logs_book, 'a') as f:
            f.write(str(self.id) + ' ' + self.status + '\n')
        
    def import_data(self, data_path='data_banknote_authentication.txt'):
        data = pd.read_csv(data_path, sep=",", header=None)
        data.columns = ["a", "b", "c", "d", 'target']
        X, y = data.iloc[:, :-1], data.iloc[:, -1:]
        #X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, \
                                                             test_size=0.2, random_state=42)
        self.status = "data loaded"
        #self.add_log()
        
    def train_model(self):
        self.model = self.model.fit(self.X_train, self.y_train.values.ravel())
        self.status = "model trained"
        self.add_log()
        #print('SAVED trained model')
        #self.save_model('./saved_models/trained')
        
    def test_model(self): 
        metric = roc_auc_score(self.y_test.values.ravel(), self.model.predict_proba(self.X_test)[:, 1])
        self.status = "model test metric: " + str(metric)
        self.add_log()
        return metric

    def save_model(self, filepath='./saved_models/'):
        joblib.dump(self.model, filepath + '.pkl')
        self.status = "model saved"
        self.add_log()
            
    def delete_saved_model(self, filepath='./saved_models/'):
        print(filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
            self.status = "model deleted" 
        self.add_log()
            
