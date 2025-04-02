"""Class for Helping ENV
"""
import os
from dotenv import load_dotenv
from app_decorator import singleton

@singleton
class EnvHelper:
    """Class for helping env variable
    """
    def __init__(self):
        load_dotenv()
        self.gather_env()

    def gather_env(self):
        """Gather all env variables
        """
        try:
            self.api_host = os.getenv("API_HOST")
            self.training_dset_path= os.getenv("COMMODITY_TRAINING")
            self.testing_dset_path = os.getenv("COMMODITY_TESTING")
            self.model_tracking_uri = os.getenv("MODEL_TRACKING_URI")
            self.commodity_training_parameter_path = os.getenv("COMMODITY_TRAINING_PARAMETER_PATH")
        except Exception as e:
            print(f'Gathering Env failed: {e}')
