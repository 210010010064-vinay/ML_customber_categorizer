import os
import sys
import shutil
from src.exception import CustomerException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact
from dataclasses import dataclass


@dataclass
class ModelPusherConfig:
    model_pusher_dir: str = r"C:\Users\Student\Desktop\Customber_Categorizer\models"
    pushed_model_file: str = "model.pkl"




class ModelPusher:
    def __init__(self, model_trainer_artifact, model_pusher_config):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            # Ensure directory exists
            os.makedirs(self.model_pusher_config.model_pusher_dir, exist_ok=True)

            # Source model file path (from trainer)
            src_model_path = self.model_trainer_artifact.trained_model_file_path

            # Destination model file path (local folder)
            dest_model_path = os.path.join(self.model_pusher_config.model_pusher_dir,self.model_pusher_config.pushed_model_file)

            # Copy model file to destination folder
            shutil.copy(src_model_path, dest_model_path)

            logging.info(f"Model saved locally at {dest_model_path}")

            model_pusher_artifact = ModelPusherArtifact(local_model_path=dest_model_path)

            logging.info(f"ModelPusherArtifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e
