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
        self.envs = ["API_HOST",
                    "COMMODITY_TRAINING",
                    "COMMODITY_TESTING",
                    "MODEL_TRACKING_URI",
                    "COMMODITY_TRAINING_PARAMETER_PATH"]
        self.all_available = self.check_envs()
        self.gather_env()
    
    def check_envs(self) -> bool:
        """Check if all the env is available

        Returns:
            bool: if true all env is available, else otherwise
        """
        for env in self.envs:
            if os.getenv(env) is None:
                print(f'Env {env} is missing')
                return False
        return True

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
