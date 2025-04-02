import mlflow
from helper import EnvHelper

class CommodityLogger:
    """Class for helping logging commodity"""
    def __init__(self):
        self.env_helper = EnvHelper()
    def log_finetuning(self):
        with mlflow.start_run(self.env_helper.model_tracking_uri):
            print('testing')
