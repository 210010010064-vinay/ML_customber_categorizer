from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from sklearn.metrics import f1_score
from src.exception import CustomerException
from src.pipeline.prediction_pipeline import CustomerData
from src.constant.training_pipeline import TARGET_COLUMN
from src.logger import logging

import sys
import pandas as pd

from src.ml.model.local_estimator import  CustomerClusterEstimatorLocal 
from dataclasses import dataclass
from typing import Optional
from src.entity.config_entity import Prediction_config

from src.utils.main_utils import MainUtils, load_numpy_array_data
from src.ml.metric import calculate_metric
from src.entity.artifact_entity import ClassificationMetricArtifact


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    changed_accuracy: float
    best_model_metric_artifact: Optional[ClassificationMetricArtifact]


def convert_test_numpy_array_to_dataframe(array):
    """Converts numpy array to dataframe"""
    prediction_config = Prediction_config().__dict__
    columns = list(prediction_config['prediction_schema']['columns'].keys())
    
    dataframe = pd.DataFrame(array, columns=columns)
    return dataframe


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.utils = MainUtils()
        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_best_model(self) -> Optional[CustomerClusterEstimatorLocal]: 
        """
        Loads the best model from a local path if present.
        """
        try:
            model_path = self.model_eval_config.best_model_path 
            
            estimator = CustomerClusterEstimatorLocal(model_path=model_path)

            if estimator.is_model_present():
                return estimator
            return None
        except Exception as e:
            raise CustomerException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            x_test = convert_test_numpy_array_to_dataframe(array=test_arr[:, :-1])
            y_test = pd.Series(test_arr[:, -1])

            trained_model = self.utils.load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            y_hat_trained_model = trained_model.predict(x_test)

            trained_model_f1_score = f1_score(y_test, y_hat_trained_model, average='weighted')
            best_model_f1_score = None
            best_model_metric_artifact = None

            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x_test)
                best_model_f1_score = f1_score(y_test, y_hat_best_model, average='weighted')
                best_model_metric_artifact = calculate_metric(best_model, x_test, y_test)

            best_model_score = best_model_f1_score or 0

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > best_model_score,
                changed_accuracy=trained_model_f1_score - best_model_score,
                best_model_metric_artifact=best_model_metric_artifact
            )

            logging.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_response = self.evaluate_model()
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                best_model_path=self.model_trainer_artifact.trained_model_file_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.changed_accuracy,
                best_model_metric_artifact=evaluate_model_response.best_model_metric_artifact
            )

            logging.info(f"Model evaluation artifact created: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e
