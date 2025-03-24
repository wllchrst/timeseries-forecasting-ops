"""Class for Helping ENV
"""
import os
from dotenv import load_dotenv

class EnvHelper:
    """Class for helping env variable
    """
    def __init__(self):
        env_loaded = load_dotenv()
        print(f'Env Loaded: {env_loaded}')

        self.gather_env()
    def gather_env(self):
        """Gather all env variables
        """
        self.api_host = os.getenv("API_HOST")
