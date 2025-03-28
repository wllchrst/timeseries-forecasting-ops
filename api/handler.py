"""Python file for handling every related api"""
import uvicorn
from helper import EnvHelper
from app_decorator import singleton
from .endpoint import app
@singleton
class APIHandler:
    """Class for handling api"""
    def __init__(self):
        self.app = app
        self.env_helper = EnvHelper()
        uvicorn.run(self.app, host=self.env_helper.api_host)
