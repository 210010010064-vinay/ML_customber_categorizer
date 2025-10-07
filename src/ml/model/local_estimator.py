import os
import joblib  # or pickle
from typing import Any

class CustomerClusterEstimatorLocal:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def is_model_present(self) -> bool:
        return os.path.exists(self.model_path)

    def load_model(self) -> Any:
        if not self.is_model_present():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if self.model is None:
            self.model = joblib.load(self.model_path)
        return self.model

    def predict(self, X):
        model = self.load_model()
        return model.predict(X)
