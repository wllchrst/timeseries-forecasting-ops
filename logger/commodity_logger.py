import mlflow
import torch.nn as nn
import traceback
from datetime import datetime
from helper import EnvHelper
from app_decorator import singleton

@singleton
class CommodityLogger:
    """Class for helping logging commodity"""
    
    def __init__(self):
        self.env_helper = EnvHelper()
        mlflow.set_tracking_uri(self.env_helper.model_tracking_uri)  # Set tracking URI once
        print(f'Tracking URI: {self.env_helper.model_tracking_uri}')

    def log_finetuning(self, parameter: dict, model: nn.Module):
        """Log Finetuning result.

        Args:
            parameter (dict): parameter that is used for finetuning the model
        """
        # Format the experiment name
        current_date = datetime.now().strftime("%d-%m-%Y")
        experiment_name = f"Finetuning TTM model with Commodity Data {current_date}"

        # Set experiment before starting a run
        mlflow.set_experiment(experiment_name)

        try:
            with mlflow.start_run():
                print(f'Connected to MLflow at {self.env_helper.model_tracking_uri}')
                
                # Log parameters
                mlflow.log_params(parameter)
        except Exception as e:
            print(f"Logging error: {e}")
            traceback.print_exc()
