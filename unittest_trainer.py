import unittest
import numpy as np
import os
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from trainer import BaseModelTrainer, LogisticRegressionTrainer

class TestLogisticRegressionTrainer(unittest.TestCase):

    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model_path = "test_logreg_model.pkl"
        self.trainer = LogisticRegressionTrainer("logreg_test")

    def test_train_model(self):
        self.trainer.train_model(self.X_train, self.y_train)
        self.assertEqual(self.trainer.get_status(), "trained")

    def test_save_and_load_model(self):
        self.trainer.train_model(self.X_train, self.y_train)
        self.trainer.save_model(self.model_path)
        self.assertEqual(self.trainer.get_status(), "saved")
        self.assertTrue(os.path.exists(self.model_path))

        loaded_model = joblib.load(self.model_path)
        self.assertIsNotNone(loaded_model)

    def test_delete_saved_model(self):
        self.trainer.train_model(self.X_train, self.y_train)
        self.trainer.save_model(self.model_path)
        self.trainer.delete_saved_model(self.model_path)
        self.assertEqual(self.trainer.get_status(), "initialized")
        self.assertFalse(os.path.exists(self.model_path))

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()