import mlflow
import torch
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
        
        # Set a fixed experiment name
        mlflow.set_experiment("Finetuning TTM Model with Commodity Data")

    def log_finetuning(self, parameter: dict, model: nn.Module, input_example: torch.Tensor):
        """Log Finetuning result.

        Args:
            parameter (dict): parameter that is used for finetuning the model
            model (nn.Module): trained PyTorch model
        """
        run_name = f"finetuning-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        try:
            with mlflow.start_run(run_name=run_name):
                print(f'Connected to MLflow at {self.env_helper.model_tracking_uri}')
                
                # Log parameters
                mlflow.log_params(parameter)

                mlflow.pytorch.log_model(model, "model", input_example=input_example)

                print("Logging success")

        except Exception as e:
            print(f"Logging error: {e}")
            traceback.print_exc()
